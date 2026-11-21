import os
import sys
import time
import base64
import json
import tempfile
import io
from PIL import Image

# Ensure Ultralytics config has write access
os.environ['YOLO_CONFIG_DIR'] = os.path.join(tempfile.gettempdir(), 'Ultralytics')
from ultralytics import YOLO
from flask import Flask, render_template, request, jsonify, send_from_directory

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_yolo_sandal.pt")

app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder=os.path.join(BASE_DIR, "static"))

# Load YOLO model once at startup
print(f"Memuat Model YOLOv8 dari: {MODEL_PATH}...")
try:
    model = YOLO(MODEL_PATH)
    print("Model YOLOv8 berhasil dimuat! Kelas target:", model.names)
except Exception as e:
    print(f"Error saat memuat model: {e}")
    model = None

@app.route("/")
def index():
    return render_template("index.html", model_loaded=(model is not None))

@app.route("/api/status")
def status():
    return jsonify({
        "status": "ready" if model is not None else "model_missing",
        "app_name": "Citra Vision",
        "classes": list(model.names.values()) if model else ["Berantakan", "Rapi"],
        "model_path": MODEL_PATH
    })

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"status": "error", "message": "Model belum dimuat"}), 500

    start_time = time.time()
    img = None

    # Option 1: File Upload (multipart/form-data)
    if "file" in request.files:
        file = request.files["file"]
        if file.filename != "":
            try:
                img = Image.open(file.stream).convert("RGB")
            except Exception as e:
                return jsonify({"status": "error", "message": f"Gagal membaca file citra: {str(e)}"}), 400

    # Option 2: Base64 JSON (from Webcam / Canvas)
    elif request.is_json:
        data = request.get_json()
        if "image" in data:
            image_data = data["image"]
            if "," in image_data:
                image_data = image_data.split(",")[1]
            try:
                decoded = base64.b64decode(image_data)
                img = Image.open(io.BytesIO(decoded)).convert("RGB")
            except Exception as e:
                return jsonify({"status": "error", "message": f"Gagal mendekode citra base64: {str(e)}"}), 400

    if img is None:
        return jsonify({"status": "error", "message": "Tidak ada data citra yang diterima"}), 400

    try:
        # Save temporary image for YOLO prediction
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            img.save(tmp_path, format="JPEG")

        # Run inference
        results = model.predict(tmp_path, verbose=False)
        inference_time_ms = round((time.time() - start_time) * 1000, 2)

        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        probs = results[0].probs
        top1_idx = int(probs.top1)
        top1_class = model.names[top1_idx]
        top1_conf = float(probs.top1conf)

        all_probs = {}
        for idx, class_name in model.names.items():
            all_probs[class_name] = round(float(probs.data[idx]), 4)

        return jsonify({
            "status": "success",
            "prediction": top1_class,
            "confidence": round(top1_conf, 4),
            "confidence_percent": round(top1_conf * 100, 2),
            "probabilities": all_probs,
            "inference_time_ms": inference_time_ms
        })

    except Exception as e:
        return jsonify({"status": "error", "message": f"Terjadi kesalahan saat inferensi: {str(e)}"}), 500

if __name__ == "__main__":
    print("=" * 60)
    print("  Citra Vision - Sistem Klasifikasi Kerapian Sandal")
    print("  Model: YOLOv8 (best_yolo_sandal.pt)")
    print("  Akses Web: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
