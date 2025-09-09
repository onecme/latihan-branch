import numpy as np
import tensorflow as tf
from picamera2 import Picamera2
import cv2

# --- Load TFLite model ---
interpreter = tf.lite.Interpreter(model_path="mobilenetv2_trash_float32.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- Init PiCamera2 ---
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (224, 224)}))
picam2.start()

# --- Ambil 1 frame dari kamera ---
frame = picam2.capture_array()   # langsung dapat numpy array (H,W,C), dtype=uint8
img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # ubah ke RGB (default OpenCV = BGR)

# --- Preprocessing sesuai training ---
img_array = img.astype(np.float32) / 255.0

# Jika training pakai mean-std (ImageNet style), lakukan normalisasi
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
img_array = (img_array - mean) / std

# --- Tambah batch dimension: NHWC (1,224,224,3) ---
img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

# --- Inference ---
interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()
pred = interpreter.get_tensor(output_details[0]['index'])

# --- Softmax & klasifikasi ---
probs = tf.nn.softmax(pred).numpy()[0]
class_labels = ["b3", "recycle", "wasteFood"]
predicted_idx = np.argmax(probs)
predicted_label = class_labels[predicted_idx]

print("Probabilitas:", probs)
print("Prediksi:", predicted_label, "(", probs[predicted_idx], ")")
