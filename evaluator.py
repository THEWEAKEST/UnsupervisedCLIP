import numpy as np
from tqdm import tqdm

class Evaluator:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def get_winoground_scores(self, scores):
        text = scores[0][0] > scores[1][0] and scores[1][1] > scores[0][1]
        image = scores[0][0] > scores[0][1] and scores[1][1] > scores[1][0]
        group = text and image
        return text, image, group

class ContrastiveEvaluator(Evaluator):
    def get_scores(self, captions, images):
        scores = [[0 for _ in range(len(images))] for _ in range(len(captions))]
        for caption_id, caption in enumerate(captions):
            for image_id, image in enumerate(images):
                input = self.processor(
                    text=caption,
                    images=image,
                    return_tensors="pt",
                    padding="max_length",
                )
                output = self.model(**input.to("cuda"))
                scores[caption_id][image_id] = output.logits_per_image.detach().cpu().item()
        return scores
