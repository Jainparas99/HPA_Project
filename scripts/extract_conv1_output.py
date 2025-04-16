import onnx
import onnxruntime as ort
import numpy as np
from onnx import helper, numpy_helper

# Load model
model_path = "models/vgg16.onnx"
model = onnx.load(model_path)

# --- STEP 1: Find the output of the first Conv layer ---
target_output = None
for node in model.graph.node:
    if node.op_type == "Conv" and "vgg0_conv0_weight" in node.input[1]:
        target_output = node.output[0]
        break

if target_output is None:
    raise ValueError("Could not find output of first Conv2D layer")

# --- STEP 2: Add that intermediate output as a new graph output ---
intermediate_output = helper.ValueInfoProto()
intermediate_output.name = target_output
model.graph.output.insert(0, intermediate_output)  # Insert as first output

# --- STEP 3: Save temporary modified model ---
temp_model_path = "models/vgg16_with_conv0_output.onnx"
onnx.save(model, temp_model_path)

# --- STEP 4: Run inference with ONNX Runtime ---
sess = ort.InferenceSession(temp_model_path)
input_name = sess.get_inputs()[0].name
outputs = sess.get_outputs()

# Input: same as before
input_tensor = np.load("models/input_tensor.npy")

# Get output of vgg0_conv0
intermediate_output_tensor = sess.run([outputs[0].name], {input_name: input_tensor})[0]
np.save("models/conv1_output.npy", intermediate_output_tensor)

print(f"Saved output of vgg0_conv0 to models/conv1_output.npy")
print(f"Shape: {intermediate_output_tensor.shape}")
