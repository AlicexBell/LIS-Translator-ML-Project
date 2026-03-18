import base64
import os
import sys
import numpy as np
import joblib
import pandas as pd
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

sys.path.insert(0, str(Path(__file__).parent / "src"))
from features import normalize_and_extract

app = Flask(__name__)

# --- SETUP MODELLO E MEDIAPIPE ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "lis_model.pkl")
TASK_PATH  = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

if not os.path.exists(TASK_PATH):
    raise FileNotFoundError(f"Manca il file: {TASK_PATH}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Manca il file: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

base_options = python.BaseOptions(model_asset_path=TASK_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.HandLandmarker.create_from_options(options)




@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "Nessuna immagine ricevuta"}), 400

    # Decodifica immagine base64 dal browser
    img_data = data["image"].split(",")[1]
    img_bytes = base64.b64decode(img_data)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"sign": None, "landmarks": []})

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        return jsonify({"sign": None, "landmarks": []})

    hand_landmarks = result.hand_landmarks[0]

    # Estrazione feature (88 valori: 63 raw + 25 geometriche)
    features = normalize_and_extract(hand_landmarks)
    df_features = pd.DataFrame([features], columns=model.feature_names_in_)
    prediction = model.predict(df_features)[0]

    # Landmark in coordinate relative (0-1) per disegnarli nel browser
    landmarks = [{"x": lm.x, "y": lm.y} for lm in hand_landmarks]

    return jsonify({"sign": prediction.upper(), "landmarks": landmarks})


if __name__ == "__main__":
    print("🚀 Apri il browser su: http://localhost:5000")
    app.run(debug=False, port=5000)
