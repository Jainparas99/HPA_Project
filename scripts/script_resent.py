#!/usr/bin/env python3
import torch
import torchvision.models as models

# Load your custom-op library
torch.ops.load_library("/home/spring2024/sk4858/MPI/HPA_Project/hpc.cpython-312-x86_64-linux-gnu.so")

# Load ResNet50 and grab its weights
resnet = models.resnet50(pretrained=True)
w = resnet.fc.weight
b = resnet.fc.bias

# Swap out fc for your op
class CustomFC(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        # we don't plan to train these, so register as buffers
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, in_features]
        print("Just added my first custom fc layer!!")
        return torch.ops.hpc.fc(x, self.weight, self.bias)


resnet.fc = CustomFC(resnet.fc.weight, resnet.fc.bias)

# Script & save
scripted = torch.jit.script(resnet.eval().cuda())
scripted.save("resnet50_custom_fc.pt")
print("Saved scripted model to resnet50_custom_fc.pt")
