import torch
# import groundingdino

print("Is CUDA available?", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("PyTorch version:", torch.__version__)
print("Available GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")

print("PyTorch compiled with CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

# Is CUDA available? True
# CUDA version: 11.8
# PyTorch version: 2.5.1+cu118
# Available GPU: NVIDIA GeForce RTX 4070 Ti
# PyTorch compiled with CUDA: 11.8
# CUDA available: True
# CUDA device count: 