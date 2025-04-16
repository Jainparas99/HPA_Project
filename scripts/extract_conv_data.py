import onnx
import onnxruntime as ort
import numpy as np
from onnx import numpy_helper

model_path = "models/vgg16.onnx"
model = onnx.load(model_path)

sess = ort.InferenceSession(model_path)
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# --- Create a dummy input ---
input_tensor = np.random.randn(1, 3, 224, 224).astype(np.float32)
output_tensor = sess.run([output_name], {input_name: input_tensor})[0]

# --- Save input and output for testing ---
np.save("models/input_tensor.npy", input_tensor)
np.save("models/expected_output.npy", output_tensor)

# --- Extract weights for the first conv layer (features.0) ---
for initializer in model.graph.initializer:
    if initializer.name == "vgg0_conv0_weight":
        weights = numpy_helper.to_array(initializer)
        np.save("models/conv1_weights.npy", weights)
    elif initializer.name == "vgg0_conv0_bias":
        bias = numpy_helper.to_array(initializer)
        np.save("models/conv1_bias.npy", bias)

print(f"Export complete! Shapes:")
print(f"Input: {input_tensor.shape}")
print(f"Weights: {weights.shape}")
print(f"Bias: {bias.shape}")
print(f"Output: {output_tensor.shape}")
