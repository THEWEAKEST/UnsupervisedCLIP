from transformers import CLIPModel, CLIPProcessor
from evaluator import ContrastiveEvaluator
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
import os
from clip_test import test

class CustomDataset(Dataset):
    def __init__(self, hf_dataset, dataset_root):
        self.images = hf_dataset[:]['image_1'] + hf_dataset[:]['image_2']
        self.text = hf_dataset[:]['caption_1'] + hf_dataset[:]['caption_2']
        self.dataset_root = dataset_root
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                               std=[0.26862954, 0.26130258, 0.27577711])])
        line = []
        for i in range(len(self.images)):
            line.append(self.transform(Image.open(self.dataset_root + self.images[i]).convert('RGB')))
        self.images = torch.stack(line, dim=0)
        #print(self.images.shape)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.text[idx]

    def get_i(self):
        return self.images
    def get_t(self):
        return self.text
    def update_i(self, imgs):
        self.images = imgs
    def update_t(self, txts):
        self.text = txts


def training(model, processor, dataloader, epochs=100, lr=0.00001, exptime=None, best=[0., 0., 0.], iter_id=-1, label=None, wandb_report=True):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #processor.to(device)

    text_sc = []
    img_sc = []
    group_sc = []
    loss_stat = []

    pre_str = ""
    iter_str = "semi"
    if iter_id >= 0:
        iter_str = f"_iter{iter_id}"
    if exptime != None:
        pre_str = str(exptime)
    if label != None:
        pre_str = label + pre_str

    for i in range(epochs):
        model.train()
        batch_loss = 0.
        for images, texts in dataloader:
            
            images = images.to(device)
            inputs = processor(text=texts, images=None, return_tensors='pt', padding='max_length').to(device)
            outputs = model(pixel_values=images, input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, return_loss=True)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = batch_loss + loss.item()
        batch_loss = batch_loss / len(dataloader)
        
        loss_stat.append(batch_loss)

        #print(f"loss at epoch {i}: {batch_loss}")
        model.eval()
        ts, imgs, gs = test(model, processor, rt=True)
        
        if wandb_report:
            return_epoch = i
            if iter_id >= 0:
                return_epoch = return_epoch + iter_id * epochs
            wandb.log({'epoch':return_epoch, 'loss':batch_loss, 'group scores':gs, 'images scores':imgs, 'texts scores':ts})

        if ts > best[0] and exptime != None:
            best[0] = ts
            try:
                os.mkdir(f'/scr2/fuzhit/clip_args/{pre_str}{iter_str}_txt/')
                model.save_pretrained(f'/scr2/fuzhit/clip_args/{pre_str}{iter_str}_txt/')
            except Exception:
                model.save_pretrained(f'/scr2/fuzhit/clip_args/{pre_str}{iter_str}_txt/')

        if imgs > best[1] and exptime != None:
            best[1] = imgs
            try:
                os.mkdir(f'/scr2/fuzhit/clip_args/{pre_str}{iter_str}_img/')
                model.save_pretrained(f'/scr2/fuzhit/clip_args/{pre_str}{iter_str}_img/')
            except Exception:
                model.save_pretrained(f'/scr2/fuzhit/clip_args/{pre_str}{iter_str}_img/')

        if gs > best[2] and exptime != None:
            best[2] = gs
            try:
                os.mkdir(f'/scr2/fuzhit/clip_args/{pre_str}{iter_str}_group/')
                model.save_pretrained(f'/scr2/fuzhit/clip_args/{pre_str}{iter_str}_group/')
            except Exception:
                model.save_pretrained(f'/scr2/fuzhit/clip_args/{pre_str}{iter_str}_group/')
        
        text_sc.append(ts)
        img_sc.append(imgs)
        group_sc.append(gs)

    return model, text_sc, img_sc, group_sc, best, loss_stat

if __name__ == '__main__':
    import wandb
    wandb.init(project="UnsupervisedClip")
    config = wandb.config
    model_name="openai/clip-vit-base-patch32"
    dataset_root="../data/color_swap/"
    from clip_test import test

    ts, imgs, gs = test(rt=True)
    model = CLIPModel.from_pretrained(model_name, device_map="cuda")
    processor = CLIPProcessor.from_pretrained(model_name, device_map="cuda")

    from datasets import load_dataset
    colorswap = load_dataset(dataset_root)
    lr = 0.00002
    epochs = 100
    batch_size = 16
    config.batch_size = batch_size
    config.lr = lr
    config.epochs = epochs

    dataloader = DataLoader(CustomDataset(colorswap['train'], dataset_root), batch_size=batch_size, shuffle=True)

    from datetime import datetime
    model, text_sc, img_sc, group_sc, _, loss_func = training(model, processor, lr=lr, dataloader=dataloader, epochs=epochs, exptime=datetime.now(), best=[0., 0., 0.])
    test(model, processor, test_labels=f"full data after finetune(lr={lr}, epochs={epochs}), time={datetime.now()}", pt=True)

    #wandb.save(f"{model_name.split('/')[-1]}+{dataset_root.split('/')[-2]}+{datetime.now()}")
    
    #from matplotlib import pyplot as plt

    #x_axis = np.linspace(0, epochs, epochs + 1)
    #plt.plot(x_axis, [ts] + text_sc, color='blue', marker='o', label='texts scores')
    #plt.plot(x_axis, [imgs] + img_sc, color='red', marker='o', label='images scores')
    #plt.plot(x_axis, [gs] + group_sc, color='green', marker='o', label='groups scores')

    #plt.legend()
    
    #plt.savefig(f"/scr2/fuzhit/results/{model_name.split('/')[-1]}+{dataset_root.split('/')[-2]}+{datetime.now()}.png")