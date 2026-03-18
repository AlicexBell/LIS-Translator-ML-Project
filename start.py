import cv2
import sys
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
import pandas as pd
import numpy as np
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from features import normalize_and_extract

# --- SETUP ---
model_path = "models/lis_model.pkl"
task_path = "hand_landmarker.task"

if not os.path.exists(task_path):
    print(f"❌ Errore: manca {task_path}")
    exit()

model = joblib.load(model_path)

# Inizializzazione MediaPipe
base_options = python.BaseOptions(model_asset_path=task_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

# --- AVVIO WEBCAM MODIFICATO ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Forza l'uso di DirectShow

# Se non funziona con 0, prova con 1
if not cap.isOpened():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Impostiamo risoluzione fissa e frame rate
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("🚀 Fotocamera in fase di avvio...")
time.sleep(2) # Dai più tempo all'hardware per inizializzarsi

# Scarta i primi frame neri finché non arriva un frame valido
for _ in range(30):
    cap.read()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        continue # Salta il frame se è vuoto invece di crashare
    
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Creazione immagine MediaPipe corretta
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    
    # Timestamp preciso
    frame_timestamp_ms = int(time.time() * 1000)
    
    # Esegui rilevamento
    result = detector.detect_for_video(mp_image, frame_timestamp_ms)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            # Predizione
            features = normalize_and_extract(hand_landmarks)
            df_features = pd.DataFrame([features], columns=model.feature_names_in_)
            prediction = model.predict(df_features)[0]

            # Disegno info e punti
            cv2.putText(frame, f"SEGNO: {prediction.upper()}", (20, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            for lm in hand_landmarks:
                ih, iw, _ = frame.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Mostra video
    cv2.imshow('Traduttore LIS Live', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()