import torch
torch.ops.load_library("hpc.cpython-312-x86_64-linux-gnu.so")
print("✅ custom op loaded!")
