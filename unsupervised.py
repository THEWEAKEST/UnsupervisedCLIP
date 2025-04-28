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
from trainclip import training, CustomDataset, mix_training
from clip_test import test
from scipy.optimize import linear_sum_assignment
from datetime import datetime
import argparse
import os

class threshold_scheduler:
    def __init__(self, threshold, linear_decay=None):
        self.cnt = 0
        self.threshold = threshold
    
    def forward(self):
        self.cnt = self.cnt + 1
        return self.cnt - 1

    def update(self, value):
        self.threshold = value

    def get(self):
        return self.threshold

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

def eliminate_supervised_data(dataset, sup_ds):
    supervised_images = sup_ds.get_i().copy()
    for i in range(len(supervised_images)):
        supervised_images[i] = supervised_images[i].split('/')[-1]
    supervised_texts = sup_ds.get_t()
    standard_images = dataset.get_i().copy()
    for i in range(len(standard_images)):
        standard_images[i] = standard_images[i].split('/')[-1]
    standard_texts = dataset.get_t()
    standard_root = dataset.dataset_root

    list_i = []
    list_t = []
    #print(standard_images[0])
    #print(supervised_images[0])

    for i in standard_images:
        if i not in supervised_images:
            list_i.append(standard_root + "images/" + i)
    for i in standard_texts:
        if i not in supervised_texts:
            list_t.append(i)
    return list_i, list_t, standard_root

def update_scheduler(scheduler, dataset, batch_size=16):
    texts = dataset.get_t()
    images_addr = dataset.get_i()

    images = [Image.open(img_addr).convert("RGB") for img_addr in images_addr]

    n = len(images)

    #print(f"len of match data: {n}")

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
    cnt = 0
    for i in range(n):
        if img_idx[i] == texts_idx[i]:
            cnt = cnt + 1

    scheduler.update(cnt / n)

def match(model, processor, dataset, batch_size=64, semi=None, scheduler=None, loss_lambda=None):
    
    if semi != None:
        images_addr, texts, _ = eliminate_supervised_data(dataset, semi)
    else:
        texts = dataset.get_t()
        images_addr = dataset.get_i()

    images = [Image.open(img_addr).convert("RGB") for img_addr in images_addr]

    n = len(images)

    print(f"len of match data: {n}")

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
    if scheduler != None:
        a = scheduler.get()
        if a <= 1.0:
            threshold = for_sort[int(len(for_sort) * (1.0 - scheduler.get()))]
        else:
            threshold = scheduler.get()
    else:
        threshold = None

    #print(f"the 1st top sim: {for_sort[0]}")
    #print(f"the 100st top sim: {for_sort[99]}")
    #print(f"the 199st top sim: {for_sort[199]}")

    for i in range(len(sim)):
        if threshold != None and sim[i] > threshold:
            new_i.append(images_addr[i])
            new_t.append(line[i])
    if threshold == None:
        new_i = images_addr
        new_t = line
    if semi != None and loss_lambda == None:
        print(f"after matching, the dataset has {len(dataset)} data.")
        new_i = semi.get_i() + new_i
        new_t = semi.get_t() + new_t
    if threshold != None:
        print(f"After threshold {threshold}, {len(new_i)} pairs of data left.")
    return NewDataset(new_i, new_t, dataset.dataset_root)

def greedy(model, processor, dataset, batch_size=64, semi=None, limited=True, loss_lambda=None, scheduler=None):
    
    if semi != None:
        images_addr, texts, _ = eliminate_supervised_data(dataset, semi)
    else:
        texts = dataset.get_t()
        images_addr = dataset.get_i()

    images = [Image.open(img_addr).convert("RGB") for img_addr in images_addr]
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
    sim_arr = []
    rel = np.array(rel)
    if scheduler != None:
        threshold = scheduler.get()
    else:
        threshold = None

    for ti in range(n):
        x, y = np.unravel_index(np.argmax(rel), rel.shape)
        if rel[x][y] < 0:
            break
        #print(f"top {ti+1} rel is {rel[x][y]}")
        if rank_match_threshold_check(threshold, rel[x][y]):
            chs_i.append(images_addr[x])
            chs_t.append(texts[y])
            if threshold != None and threshold < 1.0:
                sim_arr.append(rel[x][y])
        if limited:
            for i in range(n):
                rel[x][i] = -1
                rel[i][y] = -1
        else:
            rel[x][y] = -1
    
    if threshold != None and threshold <= 1.0:
        chs_i, chs_t = rank_match_threshold_ratio(chs_i, chs_t, sim_arr, threshold=threshold)

    if semi != None and loss_lambda == None:
        chs_i = semi.get_i() + chs_i
        chs_t = semi.get_t() + chs_t
    print(f"chose {len(chs_i)} pairs")
    return NewDataset(images=chs_i, texts=chs_t, dataset_root=dataset.dataset_root)

def rank_match_threshold_check(threshold, value):
    if threshold == None:
        return True
    if threshold <= 1.0:
        return True
    if value >= threshold:
        return True
    return False

def rank_match_threshold_ratio(images, texts, sim_array, threshold):
    sort_array = sim_array.copy()
    sort_array.sort()
    th = sort_array[int(len(sort_array) * (1.0 - threshold))]
    new_images = []
    new_texts = []
    for i in range(len(images)):
        if sim_array[i] >= th:
            new_images.append(images[i])
            new_texts.append(texts[i])

    return new_images, new_texts

def rankmatch(model, processor, dataset, batch_size=64, semi=None, loss_lambda=None, scheduler=None):
    if semi != None:
        images_addr, texts, _ = eliminate_supervised_data(dataset, semi)
    else:
        texts = dataset.get_t()
        images_addr = dataset.get_i()
    n = len(texts)
    images = [Image.open(img_addr).convert("RGB") for img_addr in images_addr]

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
    sim_array = []
    
    threshold = None

    if scheduler != None:
        threshold = scheduler.get()

    for i in range(n):
        for j in range(n):
            if rel[i][j] >= mid and rank_match_threshold_check(threshold, rel[i][j]):
                new_images.append(images_addr[i])
                new_texts.append(texts[j])
                #print(f"current rel: {rel[i][j]}")
                if threshold != None and threshold <= 1.0:
                    sim_array.append(rel[i][j])

    if threshold != None and threshold <= 1.0:
        new_images, new_texts = rank_match_threshold_ratio(new_images, new_texts, sim_array, threshold)
    ''' 
    for i in range(n):
        if rel[i][max_i_index[i]] >= mid:
            new_images.append(images_addr[i])
            new_texts.append(texts[max_i_index[i]])
        elif rel[max_t_index[i]][i] >= mid:
            new_images.append(images_addr[max_t_index[i]])
            new_texts.append(texts[i])'
    '''

    if semi != None and loss_lambda == None:
        new_images = semi.get_i() + new_images
        new_texts = semi.get_t() + new_texts
    print(f"{len(new_images)} pairs of data are chosen.")
    return NewDataset(images=new_images, texts=new_texts, dataset_root=dataset.dataset_root)
    
if __name__ == '__main__':
    wandb.init(project="UnsupervisedClip", entity="UnsupervisedCLIP")
    
    parser = argparse.ArgumentParser(description="Unsupervised CLIP")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--semi', type=bool, default=0)
    parser.add_argument('--loss_lambda', type=float, default=-1)
    parser.add_argument('--match_methods', type=str, default="li")
    parser.add_argument('--threshold_dynamics', type=int, default=0) # 0: no threshold 1: fixed threshold ratio 2: dynamics threshold 3: fixed threshold
    parser.add_argument('--threshold_value', type=float, default=0.7)
    #parser.add_argument('--unsupervised', type=bool, default=True)
    parser.add_argument('--rand_seed', type=int, default=0) # 0, 17, 23, 29

    args = parser.parse_args()

    print(f"The arguments are: {args}")

    if args.loss_lambda < 0:
        args.loss_lambda = None

    config = wandb.config
    #model_name="/scr2/fuzhit/clip_args/supervised_10epochs2025-04-13 19:32:37.232522semi_group_copy"
    #model_name="/scr2/fuzhit/clip_args/2025-03-21 01:10:25.295665semi_group_copy"
    if args.semi:
        #model_name="/scr2/fuzhit/clip_args/supervised_10epochs2025-04-17 09:49:16.095641semi_group_copy"
        model_name="/scr2/fuzhit/clip_args/supervised_10epochs2025-04-23 03:33:54.308468semi_group_copy"
    else:
        model_name="openai/clip-vit-base-patch32"
    processor_name="openai/clip-vit-base-patch32"
    dataset_root="../data/color_swap/"
    supervised_dataset_root="../data/color_swap_0.05k/"

    torch.manual_seed(args.rand_seed)
    torch.cuda.manual_seed_all(args.rand_seed)
    np.random.seed(args.rand_seed)

    lr = 0.00002
    epochs = args.epochs
    iter_num = 10
    batch_size = 64
    threshold = 37.2
    limited = True
    loss_lambda = args.loss_lambda # if loss_lambda is not None, seperate dataset into labeled DS and unlabeled DS
    semi = args.semi
    match_method = args.match_methods # use rank to build pseudo labels
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
    if semi and loss_lambda != None:
        supervised_images = supervised_dataset.get_i().copy()
        for i in range(len(supervised_images)):
            supervised_images[i] = supervised_images[i].split('/')[-1]
        supervised_texts = supervised_dataset.get_t()
        standard_images = standard_train.get_i()
        for i in range(len(standard_images)):
            standard_images[i] = standard_images[i].split('/')[-1]
        standard_texts = standard_train.get_t()
        standard_root = standard_train.dataset_root

        list_i = []
        list_t = []
        #print(standard_images[0])
        #print(supervised_images[0])

        for i in standard_images:
            if i not in supervised_images:
                list_i.append(standard_root + "images/" + i)
        for i in standard_texts:
            if i not in supervised_texts:
                list_t.append(i)
        standard_train.update_i(list_i)
        standard_train.update_t(list_t)
        print(f"len of standard_train: {len(standard_train)}")
        print(f"len of supervised_dataset: {len(supervised_dataset)}")
        print(f"len of total train dataset: {len(train_dataset)}")

    test_dataset = CustomDataset(colorswap['test'], dataset_root)
    
    st_t, st_i, st_g = test(model, processor, rt=True)

    text_scores = [st_t]
    images_scores = [st_i]
    group_scores = [st_g]
    loss_func = []

    global_best = [0., 0., 0.]

    scheduler = None

    if args.threshold_dynamics != 0:
        scheduler = threshold_scheduler(threshold=args.threshold_value)

    #linear_decay = False

    exptime = datetime.now()
    new_dataset = None
    linear_decay = True
    print(f"threshold scheduler is {scheduler}, model name is {model_name}, dataset is {dataset_root}, supervised dataset is {supervised_dataset_root}")
    if scheduler != None:
        print(f"Threshold is {scheduler.get()}")
    if semi and args.loss_lambda != None:
        u_ratio =  len(standard_train) / len(train_dataset) 
        u_batch_size = int(u_ratio * batch_size)
        l_batch_size = batch_size - u_batch_size
        print(f"labeled batch size is {l_batch_size}")
        print(f"unlabel batch size is {u_batch_size}")
    for iter in tqdm(range(iter_num)):
        model.eval()
        #test(model, processor, test_labels=f"unsupervised_iter {iter}")
        if match_method == 'rank':
            new_dataset = rankmatch(model, processor, train_dataset, semi=supervised_dataset, loss_lambda=loss_lambda, scheduler=scheduler)
        elif match_method == 'li':        
            new_dataset = match(model, processor, train_dataset, semi=supervised_dataset, scheduler=scheduler, loss_lambda=loss_lambda)
        else:
            new_dataset = greedy(model, processor, train_dataset, semi=supervised_dataset, limited=limited, loss_lambda=loss_lambda, scheduler=scheduler)
        
        new_texts = new_dataset.get_t()
        new_images = new_dataset.get_i().copy()
        for i in range(len(new_images)):
            new_images[i] = new_images[i].split('/')[-1]
        standard_images = standard_train.get_i().copy()
        for i in range(len(standard_images)):
            standard_images[i] = standard_images[i].split('/')[-1]
        standard_texts = standard_train.get_t()
        match_accuracy = 0
        for i in range(len(new_texts)):
            if standard_texts.index(new_texts[i]) == standard_images.index(new_images[i]):      
                match_accuracy = match_accuracy + 1
        wandb.log({'match accuracy': match_accuracy, "chosen": len(new_texts), "iteration": iter})

        if loss_lambda == None: # not seperate
            dataloader = DataLoader(new_dataset, batch_size=batch_size, shuffle=True)
            model, text_sc, img_sc, group_sc, best, loss_stat = training(model, processor,
                                                                     lr=lr,
                                                                     dataloader=dataloader,
                                                                     epochs=epochs,
                                                                     exptime=exptime,
                                                                     best=global_best,
                                                                     iter_id=iter,
                                                                     label='unsupervised',
                                                                     linear_decay=linear_decay)
        
        else:
            #train with labeled data
            sup_dataloader = DataLoader(supervised_dataset, batch_size=l_batch_size, shuffle=True)
            '''
            sup_dataloader = DataLoader(supervised_dataset, batch_size=batch_size, shuffle=True)
            model, text_sc, img_sc, group_sc, best, loss_stat = training(model, processor,
                                                                     lr=lr,
                                                                     dataloader=sup_dataloader,                                                                    
                                                                     epochs=epochs,
                                                                     exptime=exptime,
                                                                     best=global_best,
                                                                     iter_id=iter,
                                                                     label='unsupervised',
                                                                     factor=loss_lambda,
                                                                     linear_decay=linear_decay)
            '''
            #train with unlabeled data
            dataloader = DataLoader(new_dataset, batch_size=u_batch_size, shuffle=True)
            model, text_sc, img_sc, group_sc, best, loss_stat = mix_training(model, processor,
                                                                     lr=lr,
                                                                     dataloader=sup_dataloader,
                                                                     u_dataloader=dataloader,
                                                                     epochs=epochs,
                                                                     exptime=exptime,
                                                                     best=global_best,
                                                                     iter_id=iter,
                                                                     label='unsupervised',
                                                                     factor=loss_lambda,
                                                                     linear_decay=linear_decay)
            '''
            dataloader = DataLoader(new_dataset, batch_size=batch_size, shuffle=True)
            model, text_sc, img_sc, group_sc, best, loss_stat = training(model, processor,
                                                                     lr=lr,
                                                                     dataloader=dataloader,                                                                     
                                                                     epochs=epochs,
                                                                     exptime=exptime,
                                                                     best=global_best,
                                                                     iter_id=iter,
                                                                     label='unsupervised',
                                                                     factor=loss_lambda,
                                                                     linear_decay=linear_decay)
            '''

        if args.threshold_dynamics == 2:
            update_scheduler(scheduler, supervised_dataset)
            wandb.log({'after updating threshold: ': scheduler.get()})
        
        text_scores = text_scores + text_sc
        images_scores = images_scores + img_sc
        group_scores = group_scores + group_sc
        loss_func = loss_func + loss_stat
        for i in range(3):
            if global_best[i] < best[i]:
                global_best[i] = best[i]
    
    try:
        os.mkdir(f'/scr2/fuzhit/clip_args/{exptime}_final/')
        model.save_pretrained(f'/scr2/fuzhit/clip_args/{exptime}_final/')
    except Exception:
        model.save_pretrained(f'/scr2/fuzhit/clip_args/{exptime}_final/')
    
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
