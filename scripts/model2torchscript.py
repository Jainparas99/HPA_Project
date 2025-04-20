import torch
import torchvision.models as models
import os

# configs
model = 'resnet50'
model_save_dir = 'models'

print("*"*80)
print(' '*20, "Running the model porting script !!")
print("*"*80)
print(f"[INFO] Loading the model weights for {model} from torch hub ")
# Loading sample model from hub
resnet50_model_with_weights = torch.hub.load('pytorch/vision', model, weights='IMAGENET1K_V2')
print(f"[INFO] Successfully loaded the model")
# Save the Torch script model
sm = torch.jit.script(resnet50_model_with_weights)
save_path = os.path.join(model_save_dir, f"pytorch_{model}.pt")
sm.save(save_path)
print(f"[INFO] Successfully saved the model to {save_path}")

# Load the saved model
loaded_model = torch.jit.load("models/pytorch_resnet50.pt")

# Test inference with a sample input
sample_input = torch.rand((1,3,224,224))
print(sample_input.shape)
output = loaded_model(sample_input)
print("Inference Output Shape: ", output.shape)
print("*"*80)
print(' '*20, "Congrats! Model Ported to Torchscript")
print("*"*80)
