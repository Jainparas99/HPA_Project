import numpy as np
import cv2
import time
import onnxruntime as ort

# ---- Load and preprocess image ----
img_path = "001.jpg"  # your input
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
img = img.astype(np.float32) / 255.0
img = img.transpose(2, 0, 1)  
img = np.expand_dims(img, axis=0)  # [1, 3, 224, 224]

np.save("/home/stu15/s15/pj2196/HPA/final/HPA_Project/models/input_tensor.npy", img)  # for CUDA input

# ---- Run ONNX inference ----
session = ort.InferenceSession("/home/stu15/s15/pj2196/HPA/final/HPA_Project/build/vgg16.onnx", providers=["CUDAExecutionProvider"])
input_name = session.get_inputs()[0].name

# Warmup
for _ in range(5):
    session.run(None, {input_name: img})

# Timing
start = time.time()
for _ in range(50):  # average across runs
    onnx_out = session.run(None, {input_name: img})
end = time.time()
avg_time = (end - start) / 50 * 1000

# Save output for comparison
np.save("/home/stu15/s15/pj2196/HPA/final/HPA_Project/models/expected_output.npy", onnx_out[0])

print(f"ONNX Runtime Inference Time: {avg_time:.3f} ms")