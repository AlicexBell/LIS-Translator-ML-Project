# Overfitting perché train/test split avviene dopo l'augmentation
import pandas as pd
import joblib
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def train_lis_model(csv_path, model_output_path):
    # 1. Caricamento dati
    if not os.path.exists(csv_path):
        print(f"❌ Errore: Il file {csv_path} non esiste!")
        return

    df = pd.read_csv(csv_path)
    print(f"📊 Dataset caricato: {df.shape[0]} campioni, {df.shape[1]-1} feature.")

    # 2. Preparazione Feature (X) e Target (y)
    X = df.drop('target', axis=1)
    y = df['target']

    # 3. Split Training e Test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Creazione Modello
    print("🏋️‍♂️ Addestramento del Random Forest (300 alberi)...")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_features='sqrt',
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train, y_train)

    # 5. Valutazione
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Accuratezza Modello: {acc * 100:.2f}%")
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

    # 7. Salvataggio del modello
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(clf, model_output_path)
    print(f"💾 Modello salvato con successo in: {model_output_path}")


if __name__ == "__main__":
    _ROOT = Path(__file__).resolve().parent.parent
    CSV_INPUT = _ROOT / "data" / "processed" / "lis_landmarks.csv"
    MODEL_OUT  = _ROOT / "models" / "lis_model.pkl"
    train_lis_model(str(CSV_INPUT), str(MODEL_OUT))
