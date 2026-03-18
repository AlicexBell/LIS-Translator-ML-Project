import cv2
import os
import pandas as pd
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def extract_landmarks(base_path):
    # Setup modello (Assicurati che hand_landmarker.task sia nella root del progetto)
    model_path = "hand_landmarker.task"

    if not os.path.exists(model_path):
        print(f"❌ Errore: File {model_path} non trovato! Scaricalo con: !wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        return None

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1
    )

    detector = vision.HandLandmarker.create_from_options(options)

    data = []

    dataset_dir = os.path.join(base_path, 'LIS-fingerspelling-dataset')
    if not os.path.exists(dataset_dir):
        dataset_dir = base_path

    if not os.path.exists(dataset_dir):
        print(f"❌ Errore: Cartella {dataset_dir} non trovata!")
        return None

    classes = [c for c in sorted(os.listdir(dataset_dir)) if os.path.isdir(os.path.join(dataset_dir, c))]

    for label in classes:
        label_path = os.path.join(dataset_dir, label)
        print(f"📖 Processando classe: {label}...", end=" ")

        count = 0
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

            # Inference
            result = detector.detect(mp_image)

            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    # --- INIZIO NUOVA NORMALIZZAZIONE (TRASLAZIONE + SCALATURA) ---
                    # 1. Traslazione: Il polso è l'indice 0
                    wrist = hand_landmarks[0]
                    
                    raw_coords = []
                    for lm in hand_landmarks:
                        # Calcoliamo le coordinate relative al polso
                        raw_coords.extend([
                            lm.x - wrist.x, 
                            lm.y - wrist.y, 
                            lm.z - wrist.z
                        ])

                    # 2. Scalatura: Troviamo il valore massimo assoluto in questa mano
                    # Usiamo 1e-6 per evitare divisioni per zero se MediaPipe sbaglia
                    max_val = max(max(map(abs, raw_coords)), 1e-6)
                    
                    # Dividiamo ogni coordinata per il valore massimo
                    row = [val / max_val for val in raw_coords]
                    # --- FINE NUOVA NORMALIZZAZIONE ---
                    
                    row.append(label)
                    data.append(row)
                    count += 1

        print(f"Fatto! ({count} immagini)")

    if not data:
        return None

    cols = []
    for i in range(21):
        cols.extend([f'x{i}', f'y{i}', f'z{i}'])
    cols.append('target')

    return pd.DataFrame(data, columns=cols)