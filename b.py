import numpy as np
import tensorflow as tf
import cv2
import time
import json
import os
from datetime import datetime

# --- Load TFLite model ---
interpreter = tf.lite.Interpreter(model_path="mobilenetv2_trash_float32.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- Init Kamera OpenCV ---
cap = cv2.VideoCapture(1)

class_labels = ["b3", "recycle", "wasteFood"]

# --- Variabel tracking ---
last_label = None
label_start_time = None
hold_duration = 5  # detik

results = []  # untuk simpan hasil prediksi (list of dict)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Preprocessing ---
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_array = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # --- Inference ---
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])

    # --- Softmax & klasifikasi ---
    probs = tf.nn.softmax(pred).numpy()[0]
    predicted_idx = np.argmax(probs)
    predicted_label = class_labels[predicted_idx]
    prob = float(probs[predicted_idx])

    # --- Cek apakah label sama terus ---
    current_time = time.time()
    if predicted_label == last_label:
        if label_start_time and (current_time - label_start_time >= hold_duration):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entry = {
                "timestamp": timestamp,
                "label": predicted_label,
                "probability": prob
            }
            results.append(entry)
            print("Data disimpan:", entry)

            # reset biar tidak nyimpan terus
            label_start_time = None
            last_label = None
    else:
        last_label = predicted_label
        label_start_time = current_time

    # --- Tampilkan hasil di frame ---
    text = f"{predicted_label} ({prob:.2f})"
    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Kamera - Prediksi Sampah", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# --- Gabungkan data lama + baru ke JSON ---
filename = "prediksi.json"

if os.path.exists(filename):
    with open(filename, "r") as f:
        try:
            existing_data = json.load(f)
        except json.JSONDecodeError:
            existing_data = []
else:
    existing_data = []

# satukan hasil lama dan baru
all_results = existing_data + results

with open(filename, "w") as f:
    json.dump(all_results, f, indent=4)

print(f"Semua data berhasil disimpan ke {filename}")
