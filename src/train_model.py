import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_lis_model(csv_path, model_output_path):
    # 1. Caricamento dati
    if not os.path.exists(csv_path):
        print(f"❌ Errore: Il file {csv_path} non esiste!")
        return
    
    df = pd.read_csv(csv_path)
    print(f"📊 Dataset caricato: {df.shape[0]} campioni.")

    # 2. Preparazione Feature (X) e Target (y)
    X = df.drop('target', axis=1)
    y = df['target']

    # 3. Split Training e Test (80/20)
    # 'stratify=y' serve a mantenere le stesse proporzioni di lettere in entrambi i set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Creazione Modello
    print("🏋️‍♂️ Addestramento del Random Forest...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # 5. Valutazione veloce
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Accuratezza Modello: {acc * 100:.2f}%")
    print("\n📝 Report di Classificazione:\n", classification_report(y_test, y_pred))

    # 6. Salvataggio del modello
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(clf, model_output_path)
    print(f"💾 Modello salvato con successo in: {model_output_path}")

if __name__ == "__main__":
    # Percorsi di default
    CSV_INPUT = "/content/LIS-Translator-ML-Project/data/processed/lis_landmarks.csv"
    MODEL_OUT = "/content/LIS-Translator-ML-Project//models/lis_model.pkl"
    train_lis_model(CSV_INPUT, MODEL_OUT)