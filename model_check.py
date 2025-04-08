import torch
import pprint

# ckpt 불러오기
ckpt_path = "./training_results/cousin_ckpt.pth"
ckpt = torch.load(ckpt_path, map_location="cpu")

# 예쁘게 출력 도우미
pp = pprint.PrettyPrinter(indent=2, width=100)

# 1. 모델 파라미터 목록
print("\n📌 Model Parameters:")
for k, v in ckpt["model"].items():
    print(f"{k}: {tuple(v.shape)}")

# 2. Config 확인
print("\n📌 Config (training 설정 일부):")
pp.pprint(dict(ckpt["config"].get("train", {})))

# 3. 알고리즘 이름
print("\n📌 Algorithm Name:")
print(ckpt["algo_name"])

# 4. 환경 메타데이터
print("\n📌 Env Metadata:")
pp.pprint(ckpt["env_metadata"])

# 5. Shape Metadata (obs/action shapes 등)
print("\n📌 Shape Metadata:")
pp.pprint(ckpt["shape_metadata"])
