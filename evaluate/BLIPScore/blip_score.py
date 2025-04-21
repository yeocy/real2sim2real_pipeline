from deps.GroundingDINO.groundingdino.util.inference import preprocess_caption
import torch
from PIL import Image
import numpy as np
import os
import pathlib
import json
import matplotlib.pyplot as plt
import warnings
from packaging import version
from pathlib import Path
import argparse

from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'candidates_json',
        type=str,
        help='Candidates json mapping from image_id --> candidate.')

    parser.add_argument(
        'image_dir',
        type=str,
        help='Directory of images, with the filenames as image ids.')

    parser.add_argument(
        '--compute_other_ref_metrics',
        default=1,
        type=int,
        help='If references is specified, should we compute standard reference-based metrics?')

    parser.add_argument(
        '--save_per_instance',
        default=None,
        help='if set, we will save per instance clipscores to this file')

    args = parser.parse_args()

    if isinstance(args.save_per_instance, str) and not args.save_per_instance.endswith('.json'):
        print('if you\'re saving per-instance, please make sure the filepath ends in json.')
        quit()
    return args

def image_preprocessing(img_path, model, device):

    image = Image.open(img_path).convert("RGB")
    with torch.no_grad():
        img = model["eval"](image).unsqueeze(0).to(device)
        return img

def text_preprocessing(caption, model, device):
    with torch.no_grad():
        txt = model["eval"](caption)
        return txt


def main():
    args = parse_args()

    image_paths = [os.path.join(args.image_dir, path) for path in os.listdir(args.image_dir)
                   if path.endswith(('.png', '.jpg', '.jpeg', '.tiff'))]

    with open(args.candidates_json) as f:
        candidates = json.load(f)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == 'cpu':
        warnings.warn(
            'CLIP runs in full float32 on CPU. Results in paper were computed on GPU, which uses float16. '
            'If you\'re reporting results on CPU, please note this when you report.')

    # Load Model

    model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)
    
    blip_score_result = {}
    for path in image_paths:
        image_pre = image_preprocessing(path, vis_processors, device)
        blip_score_image = []
        for text in candidates:
            text_pre = text_preprocessing(text, text_processors, device)
            itc_score = model({"image": image_pre, "text_input": text_pre}, match_head='itc')
            blip_score_image.append(itc_score.item())
        blip_score_result[Path(path).stem] = blip_score_image
    
    with open('Blip_Scores.json', 'w') as f:
        json.dump(blip_score_result, f, indent=2)



if __name__ == '__main__':
    main()
