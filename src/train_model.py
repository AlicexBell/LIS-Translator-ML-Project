import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent))
from features import augment_landmarks, mirror_landmarks


def train_lis_model(csv_path, model_output_path):
    # 1. Caricamento dati ORIGINALI (senza augmentation)
    if not os.path.exists(csv_path):
        print(f"❌ Errore: Il file {csv_path} non esiste!")
        return

    df = pd.read_csv(csv_path)
    print(f"📊 Dataset originale: {df.shape[0]} campioni, {df.shape[1]-1} feature.")

    X = df.drop('target', axis=1)
    y = df['target']

    # 2. Split PRIMA dell'augmentation — il test set è puro (solo campioni originali)
    X_train_orig, X_test, y_train_orig, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train originale: {len(X_train_orig)} | Test (puro): {len(X_test)}")

    # 3. Augmentation SOLO sul training set
    print("⚙️  Augmentation del training set...")
    aug_rows = []
    aug_labels = []

    for i in range(len(X_train_orig)):
        coords = X_train_orig.iloc[i].tolist()
        label  = y_train_orig.iloc[i]

        # Copia specchiata
        mirrored = mirror_landmarks(coords)
        aug_rows.append(mirrored)
        aug_labels.append(label)

        # 5 varianti augmentate
        for aug in augment_landmarks(coords, n_aug=5):
            aug_rows.append(aug)
            aug_labels.append(label)

    X_aug = pd.DataFrame(aug_rows, columns=X_train_orig.columns)
    y_aug = pd.Series(aug_labels)

    X_train = pd.concat([X_train_orig, X_aug], ignore_index=True)
    y_train  = pd.concat([y_train_orig, y_aug], ignore_index=True)
    print(f"   Train dopo augmentation: {len(X_train)} campioni")

    # 4. Addestramento
    print("🏋️‍♂️ Addestramento del Random Forest (300 alberi)...")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_features='sqrt',
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train, y_train)

    # 5. Valutazione sul test set PURO (campioni originali mai visti)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Accuratezza sul test set puro: {acc * 100:.2f}%")
    print("\n📝 Report di Classificazione:\n", classification_report(y_test, y_pred))

    # 6. Confusion matrix
    out_dir = Path(model_output_path).parent.parent
    cm_path = str(out_dir / "confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(14, 12))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, xticks_rotation='vertical')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"📊 Confusion matrix salvata in: {cm_path}")

    # 7. Salvataggio
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(clf, model_output_path)
    print(f"💾 Modello salvato in: {model_output_path}")


if __name__ == "__main__":
    _ROOT = Path(__file__).resolve().parent.parent
    CSV_INPUT = _ROOT / "data" / "processed" / "lis_landmarks.csv"
    MODEL_OUT  = _ROOT / "models" / "lis_model.pkl"
    train_lis_model(str(CSV_INPUT), str(MODEL_OUT))
