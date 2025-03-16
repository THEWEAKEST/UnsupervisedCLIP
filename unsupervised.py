from transformers import CLIPModel, CLIPProcessor, TrainingArguments, Trainer
from evaluator import ContrastiveEvaluator
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import wandb
from datasets import load_dataset
from trainclip import training, CustomDataset
from clip_test import test
from scipy.optimize import linear_sum_assignment
from datetime import datetime

def match(model, processor, dataset, batch_size=256):
    images_addr = dataset.get_i()
    images = [Image.open(img_addr).convert("RGB") for img_addr in images_addr]
    texts = dataset.get_t()
    n = len(texts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rel = torch.zeros((n, n), dtype=float)

    remind = int(n % batch_size != 0)

    for i in range(int(n/batch_size) + remind):
        for j in range(int(n/batch_size) + remind):
            i_s = i * batch_size
            i_t = min(i * batch_size + batch_size, n)
            j_s = j * batch_size
            j_t = min(j * batch_size + batch_size, n)
            model.eval()
            inputs = processor(
                text=texts[j_s:j_t],
                images=images[i_s:i_t],
                return_tensors='pt',
                padding='max_length',
                )
            rel_output = model(**inputs.to(device))
            rel_output = rel_output.logits_per_image.detach()
            for x in range(i_t - i_s):
                for y in range(j_t - j_s):
                    rel[i_s + x][j_s + y] = rel_output[x][y]
    img_idx, texts_idx = linear_sum_assignment(rel, maximize=True)
    line = texts.copy()
    assert len(img_idx) == n, "length of img_idx didn't match n"
    for i in range(len(img_idx)):
        line[img_idx[i]] = texts[texts_idx[i]]
    dataset.update_t(line)
    

if __name__ == '__main__':
    wandb.init(project="UnsupervisedClip", entity="UnsupervisedCLIP")
    config = wandb.config
    model_name="openai/clip-vit-base-patch32"
    processor_name="openai/clip-vit-base-patch32"
    dataset_root="../data/color_swap/"

    lr = 0.00002
    epochs = 30
    iter_num = 10
    batch_size = 64
    config.batch_size = batch_size
    config.lr = lr
    config.epochs = epochs
    config.iter_number = iter_num

    from datasets import load_dataset
    colorswap = load_dataset(dataset_root)

    #test(test_labels="unsupervised")
    model = CLIPModel.from_pretrained(model_name, device_map="cuda")
    processor = CLIPProcessor.from_pretrained(processor_name, device_map="cuda")
    train_dataset = CustomDataset(colorswap['train'], dataset_root)
    test_dataset = CustomDataset(colorswap['test'], dataset_root)
    
    st_t, st_i, st_g = test(model, processor, rt=True)

    text_scores = [st_t]
    images_scores = [st_i]
    group_scores = [st_g]
    loss_func = []

    global_best = [0., 0., 0.]

    exptime = datetime.now()

    for iter in tqdm(range(iter_num)):
        model.eval()
        test(model, processor, test_labels=f"unsupervised_iter {iter}")
        match(model, processor, train_dataset)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        model, text_sc, img_sc, group_sc, best, loss_stat = training(model, processor, lr=lr, dataloader=dataloader, epochs=epochs, exptime=exptime, best=global_best, iter_id=iter, label='unsupervised')
        text_scores = text_scores + text_sc
        images_scores = images_scores + img_sc
        group_scores = group_scores + group_sc
        loss_func = loss_func + loss_stat
        for i in range(3):
            if global_best[i] < best[i]:
                global_best[i] = best[i]
    
    #from matplotlib import pyplot as plt

    #length = len(text_scores)

    #loss_func.append(loss_func[-1])
    #x_axis = np.linspace(0, length, length)
    #plt.plot(x_axis, text_scores, color='blue', marker='o', label='texts scores')
    #plt.plot(x_axis, images_scores, color='red', marker='o', label='images scores')
    #plt.plot(x_axis, group_scores, color='green', marker='o', label='groups scores')
    #plt.plot(x_axis, loss_func, color='black', marker='o', label='loss')

    #plt.legend()
    
    #plt.savefig(f"/scr2/fuzhit/results/unsupervised+{dataset_root.split('/')[-2]}+{datetime.now()}.png")