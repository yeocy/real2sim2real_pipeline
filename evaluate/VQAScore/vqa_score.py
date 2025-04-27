import os
import json
import t2v_metrics
from pathlib import Path

#clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl') # our recommended scoring model
key = ""

image_dir="images/"
candidates_json = "captions/captions.json"

score_func = t2v_metrics.get_score_model(model="gpt-4o", device="cuda", openai_key=key, top_logprobs=20) # We find top_logprobs=20 to be sufficient for most (image, text) samples. Consider increase this number if you get errors (the API cost will not increase).

### For a single (image, text) pair
#image = "images/image1.jpg" # an image path in string format
#text = "an orange cat and a grey cat are lying together."
images = [
    "images/image1.jpg",
    "images/image2.jpg"
]

texts = [
    "an orange cat and a grey cat are lying together.",
    "a man sitting on a bench near a lake."
]

image_paths = [os.path.join(image_dir, path) for path in os.listdir(image_dir)
                   if path.endswith(('.png', '.jpg', '.jpeg', '.tiff'))]

with open(candidates_json) as f:
    captions = json.load(f)


score = score_func(images=image_paths, texts=captions)

print("#####################################################")
print(score)

VQA_score_result = {}

for i, path in enumerate(image_paths):
    VQA_score_result[Path(path).stem] = score[i].tolist()

print(VQA_score_result)

with open('VQA_Scores.json', 'w') as f:
    json.dump(VQA_score_result, f, indent=2)


