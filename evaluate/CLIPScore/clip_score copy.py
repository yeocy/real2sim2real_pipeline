import clip
import torch
import os
import numpy as np
import warnings
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
import json
from packaging import version

def load_images(image_paths, device):
    preprocess = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])
    images = []
    for path in tqdm(image_paths, desc="Loading images"):
        image = preprocess(Image.open(path))
        images.append(image)
    return torch.stack(images).to(device)


def compute_clip_scores(model, device, images, caption, w=2.5):
    with torch.no_grad():
        if device == "cuda":
            images = images.to(torch.float16)
        else:
            images = images.to(torch.float32)

        # (1) Feature extraction
        image_features = model.encode_image(images).cpu().numpy()
        text_tokens = clip.tokenize([caption]).to(device)
        text_features = model.encode_text(text_tokens).cpu().numpy()
        text_features = np.repeat(text_features, len(image_features), axis=0)

        # (2) Normalize (same as get_clip_score)
        if version.parse(np.__version__) < version.parse('1.21'):
            image_features = sklearn.preprocessing.normalize(image_features, axis=1)
            text_features = sklearn.preprocessing.normalize(text_features, axis=1)
        else:
            warnings.warn(
                'due to a numerical instability, new numpy normalization is slightly different than paper results. '
                'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')

            image_features = image_features / np.sqrt(np.sum(image_features**2, axis=1, keepdims=True))
            text_features = text_features / np.sqrt(np.sum(text_features**2, axis=1, keepdims=True))

        # (3) Cosine similarity + clip + weight
        dot = np.sum(image_features * text_features, axis=1)
        clip_scores = w * np.clip(dot, 0, None)

        return clip_scores



def evaluate_images(image_dir, caption):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()

    image_paths = [os.path.join(image_dir, fname)
                   for fname in os.listdir(image_dir)
                   if fname.endswith(('.jpg', '.jpeg', '.png'))]

    images = load_images(image_paths, device)
    clip_scores = compute_clip_scores(model, device, images, caption)

    results = {os.path.basename(p): float(score)
               for p, score in zip(image_paths, clip_scores)}
    return results


if __name__ == "__main__":
    # 예시 caption과 이미지 폴더
    caption = "an orange cat and a grey cat are lying together."
    image_dir = "./images"  # 여기에 이미지 넣기

    scores = evaluate_images(image_dir, caption)

    print(json.dumps(scores, indent=2))
