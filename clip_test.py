from transformers import CLIPModel, CLIPProcessor
from evaluator import ContrastiveEvaluator
import numpy as np
from tqdm import tqdm
from PIL import Image

def test(model=None, processor=None, model_name="openai/clip-vit-base-patch32", dataset_root="../data/color_swap/", test_labels="normal test", rt=False, pt=False, progress=False):
    
    if model == None:
        model = CLIPModel.from_pretrained(model_name, device_map="cuda").eval()
        try:
            processor = CLIPProcessor.from_pretrained(model_name, device_map="cuda")
        except Exception:
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", device_map="cuda")

    from datasets import load_dataset

    colorswap = load_dataset(dataset_root)

    limited = 20
    count_num = 0

    evaluator = ContrastiveEvaluator(model, processor)
    text_scores, image_scores, group_scores = [], [], []
    if progress:
        pbar = tqdm(total=limited)
    for sample in colorswap["test"]:
        #if count_num == limited:
        #    break
        #print(sample)
        scores = evaluator.get_scores(
            captions=[
                sample["caption_1"],
                sample["caption_2"],
            ],
            images=[
                Image.open(dataset_root + sample["image_1"]).convert("RGB"),
                Image.open(dataset_root + sample["image_2"]).convert("RGB"),
            ],
        )
        text, image, group = evaluator.get_winoground_scores(scores)
        text_scores.append(text)
        image_scores.append(image)
        group_scores.append(group)
        #count_num = count_num + 1
        if progress:
            pbar.update(1)

    if rt:
        return np.mean(text_scores) * 100, np.mean(image_scores) * 100, np.mean(group_scores) * 100
    #print()
    
    if pt:
        with open(f"/scr2/fuzhit/results/{model_name.split('/')[-1]}_{dataset_root.split('/')[-2]}", 'a') as file:
            file.write(f"\nlabel of the test:{test_labels} \n")
            file.write("text score:  {:.2f} \n".format(np.mean(text_scores) * 100))
            file.write("image score: {:.2f} \n".format(np.mean(image_scores) * 100))
            file.write("group score: {:.2f}".format(np.mean(group_scores) * 100))
            print(f"model:{model_name}")
            print("text score:  {:.2f}".format(np.mean(text_scores) * 100))
            print("image score: {:.2f}".format(np.mean(image_scores) * 100))
            print("group score: {:.2f}".format(np.mean(group_scores) * 100))

if __name__ == '__main__':
    test(model=None, processor=None, model_name="/scr2/fuzhit/clip_args/unsupervised2025-03-04 00:04:54.693360_iter18_group", pt=True)
    '''
        openai/clip-vit-base-patch32
        /scr2/fuzhit/clip_args/2025-03-03 02:35:49.556610
        /scr2/fuzhit/clip_args/2025-03-02 10:12:04.
        /scr2/fuzhit/clip_args/unsupervised2025-03-04 00:04:54.693360_iter18_group
    '''