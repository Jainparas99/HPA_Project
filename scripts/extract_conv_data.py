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
    name = initializer.name
    # We're looking for conv layer weights and biases
    if "vgg0_conv" in name:
        data = numpy_helper.to_array(initializer)

        if "weight" in name:
            layer_num = name.split("vgg0_conv")[1].split("_")[0]
            np.save(f"models/conv{layer_num}_weights.npy", data)
            print(f"Saved weights for conv{layer_num}: {data.shape}")

        elif "bias" in name:
            layer_num = name.split("vgg0_conv")[1].split("_")[0]
            np.save(f"models/conv{layer_num}_bias.npy", data)
            print(f"Saved bias for conv{layer_num}: {data.shape}")

print(f"Export complete! Shapes:")
print(f"Input: {input_tensor.shape}")
print(f"Output: {output_tensor.shape}")
