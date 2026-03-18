# LIS-Translator-ML-Project

Traduttore in tempo reale dell'alfabeto LIS (Lingua dei Segni Italiana) tramite riconoscimento di gesture della mano, usando MediaPipe e un classificatore Random Forest.

## Stack tecnologico

- **MediaPipe HandLandmarker** — rilevamento dei 21 keypoint della mano
- **scikit-learn RandomForestClassifier** — classificazione dei segni
- **OpenCV** — acquisizione video da webcam

## Struttura del progetto

```
LIS-Translator-ML-Project/
├── data/
│   └── processed/
│       └── lis_landmarks.csv      # Dataset landmark estratti
├── models/
│   └── lis_model.pkl              # Modello addestrato
├── src/
│   ├── extractor.py               # Estrazione landmark da immagini
│   ├── train_model.py             # Addestramento e salvataggio modello
│   └── predict_webcam.py          # Helper di predizione
├── start.py                       # Entry point: traduttore webcam live
├── hand_landmarker.task           # Modello MediaPipe (richiesto)
└── requirements.txt
```

## Setup

1. Clona la repository
2. Installa le dipendenze:
   ```bash
   pip install -r requirements.txt
   ```
3. Assicurati che `hand_landmarker.task` sia nella cartella radice del progetto.

## Utilizzo

**Tutti i comandi vanno eseguiti dalla cartella radice del progetto.**

### 1. Estrai i landmark dal dataset di immagini
```bash
python src/extractor.py
```

### 2. Addestra il modello
```bash
python src/train_model.py
```

### 3. Avvia il traduttore live
```bash
python start.py
```
Premi `Q` per uscire.

## Google Colab

`train_model.py` usa `pathlib.Path(__file__)` per risolvere i percorsi relativamente allo script, quindi funziona sia in locale che su Colab senza modifiche.
