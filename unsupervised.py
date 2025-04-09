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

class NewDataset(Dataset):
    def __init__(self, images, texts, dataset_root):
        self.images = images
        self.texts = texts
        self.dataset_root = dataset_root
    def __getitem__(self, idx):
        return self.images[idx], self.texts[idx]
    def update_i(self, imgs):
        self.images = imgs
    def update_t(self, txts):
        self.text = txts
    def get_i(self):
        return self.images
    def get_t(self):
        return self.texts
    def __len__(self):
        return len(self.images)

def match(model, processor, dataset, batch_size=64, semi=None, threshold=None):
    images_addr = dataset.get_i()
    if semi != None:
        supervised_len = len(semi)
        images_addr = images_addr[supervised_len:]
    images = [Image.open(img_addr).convert("RGB") for img_addr in images_addr]
    texts = dataset.get_t()
    if semi != None:
        texts = texts[supervised_len:]
    n = len(texts)
    if semi != None:
        print(f"number of images: {len(images)}\nnumber of texts: {len(texts)}")

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
    sim = [0] * len(line)
    assert len(img_idx) == n, "length of img_idx didn't match n"
    for i in range(len(img_idx)):
        line[img_idx[i]] = texts[texts_idx[i]]
        sim[img_idx[i]] = rel[img_idx[i]][texts_idx[i]]
    
    if semi != None:
        dataset.update_t(semi.get_t()+line)
    else:
        dataset.update_t(line)
    new_i = []
    new_t = []
    
    # use ratio as threshold
    for_sort = sim.copy()
    for_sort.sort()

    threshold = for_sort[int(len(for_sort) * 0.5)]

    print(f"the 1st top sim: {for_sort[0]}")
    print(f"the 100st top sim: {for_sort[99]}")
    print(f"the 199st top sim: {for_sort[199]}")

    for i in range(len(sim)):
        if threshold != None and sim[i] > threshold:
            new_i.append(images_addr[i])
            new_t.append(line[i])
    if threshold == None:
        new_i = images_addr
        new_t = line
    if semi != None:
        print(f"after matching, the dataset has {len(dataset)} data.")
        new_i = semi.get_i() + new_i
        new_t = semi.get_t() + new_t
    if threshold != None:
        print(f"After threshold operation, {len(new_i)} pairs of data left.")
    return NewDataset(new_i, new_t, dataset.dataset_root)

def greedy(model, processor, dataset, batch_size=64, semi=None, limited=False):
    images_addr = dataset.get_i()
    if semi != None:
        supervised_len = len(semi)
        images_addr = images_addr[supervised_len:]
    images = [Image.open(img_addr).convert("RGB") for img_addr in images_addr]
    texts = dataset.get_t()
    if semi != None:
        texts = texts[supervised_len:]
    n = len(texts)
    if semi != None:
        print(f"number of images: {len(images)}\nnumber of texts: {len(texts)}")

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

    chs_i = []
    chs_t = []
    rel = np.array(rel)
    for ti in range(n):
        x, y = np.unravel_index(np.argmax(rel), rel.shape)
        if rel[x][y] < 0:
            break
        print(f"top {ti+1} rel is {rel[x][y]}")
        chs_i.append(images_addr[x])
        chs_t.append(texts[y])
        if limited:
            for i in range(n):
                rel[x][i] = -1
                rel[i][y] = -1
        else:
            rel[x][y] = -1
    if semi != None:
        chs_i = semi.get_i() + chs_i
        chs_t = semi.get_t() + chs_t
    print(f"chose {len(chs_i)} pairs")
    return NewDataset(images=chs_i, texts=chs_t, dataset_root=dataset.dataset_root)

def rankmatch(model, processor, dataset, batch_size=64, semi=None):
    images_addr = dataset.get_i()
    if semi != None:
        supervised_len = len(semi)
        images_addr = images_addr[supervised_len:]
    images = [Image.open(img_addr).convert("RGB") for img_addr in images_addr]
    texts = dataset.get_t()
    if semi != None:
        texts = texts[supervised_len:]
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
    sort = []
    max_i_index = []
    max_t_index = []
    for i in range(n):
        sort.append(rel[i].max().item())
        max_i_index.append(int(rel[i].argmax()))
        sort.append(rel[:,i].max().item())
        max_t_index.append(int(rel[:,i].argmax()))
    sort.sort()
    mid = sort[n]
    new_images = []
    new_texts = []
    
    for i in range(n):
        for j in range(n):
            if rel[i][j] >= mid:
                new_images.append(images_addr[i])
                new_texts.append(texts[j])
    ''' 
    for i in range(n):
        if rel[i][max_i_index[i]] >= mid:
            new_images.append(images_addr[i])
            new_texts.append(texts[max_i_index[i]])
        elif rel[max_t_index[i]][i] >= mid:
            new_images.append(images_addr[max_t_index[i]])
            new_texts.append(texts[i])'
    '''
    if semi != None:
        new_images = semi.get_i() + new_images
        new_texts = semi.get_t() + new_texts
    print(f"{len(new_images)} pairs of data are chosen.")
    return NewDataset(images=new_images, texts=new_texts, dataset_root=dataset.dataset_root)
    
if __name__ == '__main__':
    wandb.init(project="UnsupervisedClip", entity="UnsupervisedCLIP")
    config = wandb.config
    model_name="/scr2/fuzhit/clip_args/2025-03-21 01:10:25.295665semi_group_copy"
    #model_name="openai/clip-vit-base-patch32"
    processor_name="openai/clip-vit-base-patch32"
    dataset_root="../data/color_swap/"
    supervised_dataset_root="../data/color_swap_0.1k/"

    lr = 0.00002
    epochs = 20
    iter_num = 10
    batch_size = 64
    threshold = 37.2
    limited = False
    semi = True
    match_method = 'li' # use rank to build pseudo labels
    config.batch_size = batch_size
    config.lr = lr
    config.epochs = epochs
    config.iter_number = iter_num

    from datasets import load_dataset
    colorswap = load_dataset(dataset_root)
    if semi:
        supervised_dataset = CustomDataset(load_dataset(supervised_dataset_root)['train'], dataset_root=supervised_dataset_root)
    else:
        supervised_dataset = None

    #test(test_labels="unsupervised")
    model = CLIPModel.from_pretrained(model_name, device_map="cuda")
    processor = CLIPProcessor.from_pretrained(processor_name, device_map="cuda")
    train_dataset = CustomDataset(colorswap['train'], dataset_root)
    standard_train = CustomDataset(colorswap['train'], dataset_root)
    test_dataset = CustomDataset(colorswap['test'], dataset_root)
    
    st_t, st_i, st_g = test(model, processor, rt=True)

    text_scores = [st_t]
    images_scores = [st_i]
    group_scores = [st_g]
    loss_func = []

    global_best = [0., 0., 0.]

    exptime = datetime.now()
    new_dataset = None
    print(f"Match method is {match_method}, threshold is {threshold}, semi setting is {semi}, model name is {model_name}")
    for iter in tqdm(range(iter_num)):
        model.eval()
        #test(model, processor, test_labels=f"unsupervised_iter {iter}")
        if match_method == 'rank':
            new_dataset = rankmatch(model, processor, train_dataset, semi=supervised_dataset)
        elif match_method == 'li':        
            new_dataset = match(model, processor, train_dataset, semi=supervised_dataset, threshold=threshold)
            new_texts = new_dataset.get_t()
            standard_texts = standard_train.get_t()
            match_accuracy = 0
            if len(new_texts) == len(standard_texts):
                for i in range(len(new_texts)):
                    if new_texts[i] == standard_texts[i]:
                        match_accuracy = match_accuracy + 1
                wandb.log({'match accuracy': match_accuracy, "iteration": iter})
        else:
            new_dataset = greedy(model, processor, train_dataset, semi=supervised_dataset, limited=limited)
        dataloader = DataLoader(new_dataset, batch_size=batch_size, shuffle=True)
        model, text_sc, img_sc, group_sc, best, loss_stat = training(model, processor,
                                                                     lr=lr,
                                                                     dataloader=dataloader,
                                                                     epochs=epochs,
                                                                     exptime=exptime,
                                                                     best=global_best,
                                                                     iter_id=iter,
                                                                     label='unsupervised')
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
    
    #plt.savefig(f"/scr2/fuzhit/results/unsupervised+{dataset_root.split('/')[-2]}+{datetime.now()}.png")'
