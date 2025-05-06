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
import wandb
import argparse
from torch.optim.lr_scheduler import LambdaLR
def lr_lambda(current_step):
    #return 1.0 - float(current_step) / float(2100)
    return max(1.0 - float(current_step) / float(2100), 0)
class CustomDataset(Dataset):
    def __init__(self, hf_dataset, dataset_root):
        self.images = hf_dataset[:]['image_1'] + hf_dataset[:]['image_2']
        for i in range(len(self.images)):
            self.images[i] = dataset_root + self.images[i]
        '''
        images_1 = hf_dataset[:]["image_1"]
        images_2 = hf_dataset[:]["image_2"]
        self.images = [img for pair in zip(images_1, images_2) for img in pair]
        
        for i in range(len(self.images)):
            self.images[i] = dataset_root + self.images[i]
        texts_1=hf_dataset[:]['caption_1']
        texts_2=hf_dataset[:]['caption_2']
        self.text=[txt for pair in zip(texts_1, texts_2) for txt in pair]
        '''
        self.text = hf_dataset[:]['caption_1'] + hf_dataset[:]['caption_2']
        self.dataset_root = dataset_root
        #self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                               #std=[0.26862954, 0.26130258, 0.27577711])])
        #line = []
        #for i in range(len(self.images)):
        #    line.append(self.transform(Image.open(self.dataset_root + self.images[i]).convert('RGB')))
        #self.images = torch.stack(line, dim=0)
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

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

def training(model, processor, dataloader, optimizer, scheduler=None, epochs=100, exptime=None, best=[0., 0., 0.], iter_id=-1, label=None, wandb_report=True, factor=1.0, decay_f=0):
    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=weight_decay)
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    #if linear_decay:
    #    scheduler = LambdaLR(optimizer, lr_lambda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    try:
        print(f"At iteration {iter_id}, the learning rate is {optimizer.param_groups[0]['lr']}")
    except Exception:
        pass
    #print(lr,weight_decay,epochs)
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

    for i in tqdm(range(epochs)):
        model.train()
        batch_loss = 0.
        for images_addr, texts in dataloader:
            
            #images = images.to(device)
            #inputs = processor(text=texts, images=None, return_tensors='pt', padding='max_length').to(device)
            #outputs = model(pixel_values=images, input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, return_loss=True)
            images = [Image.open(img_addr).convert("RGB") for img_addr in images_addr]
            input = processor(
                    text=texts,
                    images=images,
                    return_tensors="pt",
                    padding="max_length",
                )
            outputs = model(**input.to(device), return_loss=True)
            loss = factor * outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler != None and decay_f == 0:
                scheduler.step()
        if scheduler != None and decay_f == 1:
            scheduler.step()

            batch_loss = batch_loss + loss.item()
        batch_loss = batch_loss / len(dataloader)
        
        loss_stat.append(batch_loss)

        #print(f"loss at epoch {i}: {batch_loss}")
        #if i%10==0:
        model.eval()
        ts, imgs, gs = test(model, processor, rt=True)
        #    print('middle-metrics: ')
        #print(ts,imgs,gs)
        #if i==epochs-1:
        #    model.eval()
        #    ts, imgs, gs = test(model, processor, rt=True)
        #    print('Testing-metrics: ')
        #    print(ts,imgs,gs)
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
    
    parser = argparse.ArgumentParser(description="Supervised CLIP")

    parser.add_argument('--dataset_root', type=str, default="../data/color_swap/")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--decay_f', type=int, default=0) # if decay_f is 0, decay after each batch, otherwise, 1, decay after each epoch

    args = parser.parse_args()

    wandb.init(project="UnsupervisedClip", entity="UnsupervisedCLIP")
    config = wandb.config
    model_name="openai/clip-vit-base-patch32"
    dataset_root = args.dataset_root
    #dataset_root="../data/color_swap_0.1k/"
    from clip_test import test

    ts, imgs, gs = test(rt=True)
    model = CLIPModel.from_pretrained(model_name, device_map="cuda")
    processor = CLIPProcessor.from_pretrained(model_name, device_map="cuda")

    from datasets import load_dataset
    colorswap = load_dataset(dataset_root)
    lr = 0.00002
    epochs = args.epochs
    batch_size = 70
    config.batch_size = batch_size
    config.lr = lr
    config.epochs = epochs

    weight_decay = 0.1

    dataloader = DataLoader(CustomDataset(colorswap['train'], dataset_root), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda)
    #print(len(dataloader))
    from datetime import datetime
    model, text_sc, img_sc, group_sc, _, loss_func = training(model, processor,
                                                                dataloader=dataloader,
                                                                optimizer=optimizer,
                                                                scheduler=scheduler,
                                                                epochs=epochs,
                                                                exptime=datetime.now(),
                                                                best=[0., 0., 0.],
                                                                label="supervised_10epochs",
                                                                decay_f=args.decay_f)
    '''
    model, text_sc, img_sc, group_sc, best, loss_stat = training(model, processor,
                                                                     dataloader=dataloader,
                                                                     optimizer=optimizer,
                                                                     scheduler=scheduler,                                                                     
                                                                     epochs=epochs,
                                                                     exptime=exptime,
                                                                     best=global_best,
                                                                     iter_id=iter,
                                                                     label='unsupervised',
                                                                     factor=loss_lambda,
                                                                     decay_f=args.decay_f)
    #test(model, processor, test_labels=f"full data after finetune(lr={lr}, epochs={epochs}), time={datetime.now()}", pt=True)

    '''
    #wandb.save(f"{model_name.split('/')[-1]}+{dataset_root.split('/')[-2]}+{datetime.now()}")
    
    #from matplotlib import pyplot as plt

    #x_axis = np.linspace(0, epochs, epochs + 1)
    #plt.plot(x_axis, [ts] + text_sc, color='blue', marker='o', label='texts scores')
    #plt.plot(x_axis, [imgs] + img_sc, color='red', marker='o', label='images scores')
    #plt.plot(x_axis, [gs] + group_sc, color='green', marker='o', label='groups scores')

    #plt.legend()
    
    #plt.savefig(f"/scr2/fuzhit/results/{model_name.split('/')[-1]}+{dataset_root.split('/')[-2]}+{datetime.now()}.png")

def mix_training(model, processor, dataloader, u_dataloader, optimizer, scheduler=None, epochs=100, exptime=None, best=[0., 0., 0.], iter_id=-1, label=None, wandb_report=True, factor=1.0, decay_f=0):
    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=weight_decay)
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    #if linear_decay:
    #    scheduler = LambdaLR(optimizer, lr_lambda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #print(lr,weight_decay,epochs)
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

    for i in tqdm(range(epochs)):
        model.train()
        batch_loss = 0.
        for (images_addr, texts), (u_images_addr, u_texts) in zip(dataloader, u_dataloader):
            
            #images = images.to(device)
            #inputs = processor(text=texts, images=None, return_tensors='pt', padding='max_length').to(device)
            #outputs = model(pixel_values=images, input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, return_loss=True)
            len_label = len(images_addr)
            new_images = images_addr + u_images_addr
            new_texts = texts + u_texts

            total_len = len(new_images)

            images = [Image.open(img_addr).convert("RGB") for img_addr in new_images]
            input = processor(
                    text=new_texts,
                    images=images,
                    return_tensors="pt",
                    padding="max_length",
                )
            outputs = model(**input.to(device), return_loss=True)

            label_logit = outputs.logits_per_text[:len_label, :len_label]
            u_logit = outputs.logits_per_text[len_label:, len_label:]

            #if iter_id == 0 and i == 0:
            #    print(f"shape of labeled logits: {label_logit.shape}")
            #    print(f"shape of unlabeled logits: {u_logit.shape}")
            
            loss_label = 0.
            loss_u = 0.

            if len_label > 0:
                loss_label = clip_loss(label_logit)
            if total_len > len_label:
                loss_u = clip_loss(u_logit)

            loss = loss_label + factor * loss_u

            #loss = factor * outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler != None and decay_f == 0: # decay after each batch
                scheduler.step()

            batch_loss = batch_loss + loss.item()

        if scheduler != None and decay_f == 1: #decay after each epoch
            scheduler.step()
        batch_loss = batch_loss / len(dataloader)
        
        loss_stat.append(batch_loss)

        #print(f"loss at epoch {i}: {batch_loss}")
        #if i%10==0:
        model.eval()
        ts, imgs, gs = test(model, processor, rt=True)
        #    print('middle-metrics: ')
        #print(ts,imgs,gs)
        #if i==epochs-1:
        #    model.eval()
        #    ts, imgs, gs = test(model, processor, rt=True)
        #    print('Testing-metrics: ')
        #    print(ts,imgs,gs)
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

def mix_training_ver_2(model, processor, dataloader, u_dataloader, optimizer, scheduler=None, epochs=100, exptime=None, best=[0., 0., 0.], iter_id=-1, label=None, wandb_report=True, factor=1.0, decay_f=0):
    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=weight_decay)
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    #if linear_decay:
    #    scheduler = LambdaLR(optimizer, lr_lambda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #print(lr,weight_decay,epochs)
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

    u_dataloader_iter = iter(u_dataloader)

    for i in tqdm(range(epochs)):
        model.train()
        batch_loss = 0.
        dataloader_iter = iter(dataloader)
        if len(list(u_dataloader_iter)) == 0: # if u_dataloader_iter bingo
            u_dataloader_iter = iter(u_dataloader) # reload

        #for (images_addr, texts), (u_images_addr, u_texts) in zip(dataloader, u_dataloader):
        iter_cnt = 0
        while True:
            try:
                images_addr, texts = next(dataloader_iter)
                u_images_addr, u_texts = next(u_dataloader_iter)
            except StopIteration:
                break   
            iter_cnt = iter_cnt + 1
            #images = images.to(device)
            #inputs = processor(text=texts, images=None, return_tensors='pt', padding='max_length').to(device)
            #outputs = model(pixel_values=images, input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, return_loss=True)
            len_label = len(images_addr)
            new_images = images_addr + u_images_addr
            new_texts = texts + u_texts

            total_len = len(new_images)

            images = [Image.open(img_addr).convert("RGB") for img_addr in new_images]
            input = processor(
                    text=new_texts,
                    images=images,
                    return_tensors="pt",
                    padding="max_length",
                )
            outputs = model(**input.to(device), return_loss=True)

            label_logit = outputs.logits_per_text[:len_label, :len_label]
            u_logit = outputs.logits_per_text[len_label:, len_label:]

            #if iter_id == 0 and i == 0:
            #    print(f"shape of labeled logits: {label_logit.shape}")
            #    print(f"shape of unlabeled logits: {u_logit.shape}")
            
            loss_label = 0.
            loss_u = 0.

            if len_label > 0:
                loss_label = clip_loss(label_logit)
            if total_len > len_label:
                loss_u = clip_loss(u_logit)

            loss = loss_label + factor * loss_u

            #loss = factor * outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler != None and decay_f == 0: # decay after each batch
                scheduler.step()

            batch_loss = batch_loss + loss.item()

        print(f"iter count is {iter_cnt}")
        if scheduler != None and decay_f == 1: #decay after each epoch
            scheduler.step()
        batch_loss = batch_loss / len(dataloader)
        
        loss_stat.append(batch_loss)

        #print(f"loss at epoch {i}: {batch_loss}")
        #if i%10==0:
        model.eval()
        ts, imgs, gs = test(model, processor, rt=True)
        #    print('middle-metrics: ')
        #print(ts,imgs,gs)
        #if i==epochs-1:
        #    model.eval()
        #    ts, imgs, gs = test(model, processor, rt=True)
        #    print('Testing-metrics: ')
        #    print(ts,imgs,gs)
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