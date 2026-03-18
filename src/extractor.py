import cv2
import os
import sys
import pandas as pd
import mediapipe as mp

from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Importa il modulo condiviso di feature
sys.path.insert(0, str(Path(__file__).parent))
from features import normalize_and_extract, augment_landmarks, mirror_landmarks, FEATURE_COLUMNS


def extract_landmarks(base_path):
    # Risolvi il percorso di hand_landmarker.task rispetto alla root del progetto
    root = Path(__file__).resolve().parent.parent
    model_path = str(root / "hand_landmarker.task")

    if not os.path.exists(model_path):
        print(f"❌ Errore: File {model_path} non trovato!")
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

            result = detector.detect(mp_image)

            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    # Estrai 88 feature (63 raw + 25 geometriche)
                    features = normalize_and_extract(hand_landmarks)

                    # Campione originale
                    data.append(features + [label])
                    count += 1

                    # Campione specchiato
                    mirrored = mirror_landmarks(features)
                    data.append(mirrored + [label])
                    count += 1

                    # 5 varianti augmentate dell'originale
                    for aug in augment_landmarks(features, n_aug=5):
                        data.append(aug + [label])
                        count += 1

        print(f"Fatto! ({count} campioni totali incluso augmentation)")

    if not data:
        return None

    cols = FEATURE_COLUMNS + ['target']
    return pd.DataFrame(data, columns=cols)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    dataset_path = str(root / "data" / "raw_images")

    print(f"📂 Estrazione da: {dataset_path}")
    df = extract_landmarks(dataset_path)

    if df is not None:
        out_path = root / "data" / "processed" / "lis_landmarks.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(out_path), index=False)
        print(f"✅ Salvato: {out_path}  ({df.shape[0]} righe × {df.shape[1]} colonne)")
    else:
        print("❌ Nessun dato estratto.")
