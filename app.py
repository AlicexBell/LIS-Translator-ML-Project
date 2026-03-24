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
import argparse
import json
import tensorflow as tf
from tensorflow import keras

# --- INIZIALIZZAZIONE APP e ARGOMENTI ---
app = Flask(__name__)

# Aggiungi il percorso 'src' per importare 'features'
sys.path.insert(0, str(Path(__file__).parent / "src"))
from features import normalize_and_extract

# Argomenti da riga di comando per lo switch del modello
parser = argparse.ArgumentParser(description="Avvia il server Flask per il traduttore LIS.")
parser.add_argument(
    '--model', 
    type=str,
    choices=['rf', 'cnn'],
    required=True,
    help="Scegli il modello da usare: 'rf' (Random Forest) o 'cnn' (Rete Neurale)."
)
args = parser.parse_args()
MODEL_TYPE = args.model

# --- SETUP GLOBALE MODELLI ---

# Paths
TASK_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
if not os.path.exists(TASK_PATH):
    raise FileNotFoundError(f"Manca il file: {TASK_PATH}")

RF_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "lis_model.pkl")
CNN_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "lis_model.h5")
CNN_CLASSES_PATH = os.path.join(os.path.dirname(__file__), "models", "class_names.json")

# Variabili globali
detector = None
model = None
cnn_class_names = None

# Inizializzazione del detector MediaPipe (usato da entrambi i modelli per i landmarks)
base_options = python.BaseOptions(model_asset_path=TASK_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.HandLandmarker.create_from_options(options)
print(f"INFO: MediaPipe HandLandmarker caricato.")

# Caricamento del modello scelto
if MODEL_TYPE == 'rf':
    print("INFO: Caricamento del modello Random Forest...")
    if not os.path.exists(RF_MODEL_PATH):
        raise FileNotFoundError(f"Manca il file del modello RF: {RF_MODEL_PATH}")
    model = joblib.load(RF_MODEL_PATH)
    print("INFO: Modello Random Forest caricato.")

elif MODEL_TYPE == 'cnn':
    print("INFO: Caricamento del modello CNN...")
    if not os.path.exists(CNN_MODEL_PATH):
        raise FileNotFoundError(f"Manca il file del modello CNN: {CNN_MODEL_PATH}")
    if not os.path.exists(CNN_CLASSES_PATH):
        raise FileNotFoundError(f"Manca il file delle classi CNN: {CNN_CLASSES_PATH}")
    
    # Sopprimi i warning di TensorFlow su custom objects
    model = keras.models.load_model(CNN_MODEL_PATH, compile=False)
    with open(CNN_CLASSES_PATH, 'r') as f:
        cnn_class_names = json.load(f)
    print("INFO: Modello CNN e nomi delle classi caricati.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "Nessuna immagine ricevuta"}), 400

    # --- DECODIFICA IMMAGINE ---
    img_data = data["image"].split(",")[1]
    img_bytes = base64.b64decode(img_data)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"sign": None, "landmarks": []})

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    # --- RILEVAMENTO LANDMARKS (COMUNE) ---
    # Lo facciamo sempre, perché anche in modalità CNN vogliamo visualizzarli
    result = detector.detect(mp_image)
    
    if not result.hand_landmarks:
        return jsonify({"sign": None, "landmarks": []})

    hand_landmarks = result.hand_landmarks[0]
    landmarks_for_drawing = [{"x": lm.x, "y": lm.y} for lm in hand_landmarks]
    prediction = "N/A"

    # --- LOGICA DI PREDIZIONE SPECIFICA PER MODELLO ---

    # A. Modalità CNN
    if MODEL_TYPE == 'cnn':
        # Preprocessing per la CNN
        img_resized = tf.image.resize(img_rgb, [224, 224])
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_array = tf.expand_dims(img_array, 0) # Crea un batch
        
        # Predizione
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        prediction = cnn_class_names[predicted_index]

    # B. Modalità Random Forest
    elif MODEL_TYPE == 'rf':
        features = normalize_and_extract(hand_landmarks)
        df_features = pd.DataFrame([features], columns=model.feature_names_in_)
        prediction = model.predict(df_features)[0]

    return jsonify({"sign": prediction.upper(), "landmarks": landmarks_for_drawing})


if __name__ == "__main__":
    print("==============================================")
    print(f"🚀 Avvio server in modalità: {MODEL_TYPE.upper()}")
    print("   Usa 'python app.py --model cnn' o 'python app.py --model rf'")
    print("   Apri il browser su: http://localhost:5000")
    print("==============================================")
    app.run(debug=False, port=5000)
