# Analisi del Progetto LIS Translator

Questo documento analizza la struttura, le funzionalità e i componenti chiave del progetto LIS Translator, con l'obiettivo di fornire una guida chiara per la manutenzione e lo sviluppo futuro.

## 1. Descrizione del Progetto

**LIS Translator** è un'applicazione web in tempo reale che traduce la Lingua dei Segni Italiana (LIS). L'applicazione utilizza la webcam per catturare i gesti dell'utente, li analizza tramite un modello di visione artificiale e li traduce nella parola o frase corrispondente.

Le funzionalità principali sono:

- **Riconoscimento in tempo reale:** Analisi del flusso video dalla webcam per un feedback immediato.
- **Interfaccia Web:** Una semplice interfaccia web mostra il video dell'utente, il gesto riconosciuto e i punti di riferimento (landmarks) della mano.
- **Modello di Machine Learning:** Un classificatore `RandomForestClassifier` addestrato per riconoscere i segni della LIS basandosi sui landmarks della mano.

---

## 2. Struttura del Progetto

La repository è organizzata in modo da separare i dati, i modelli, il codice sorgente e l'applicazione web.

```
c:.
├───app.py                # Applicazione principale Flask
├───start.py              # Script alternativo per avvio (probabilmente obsoleto)
├───requirements.txt      # Dipendenze Python
├───data/
│   └───processed/
│       └───lis_landmarks.csv # Dataset per l'addestramento
├───models/
│   ├───lis_model.pkl       # Modello di ML pre-addestrato
│   └───hand_landmarker.task  # Modello di MediaPipe
├───src/
│   ├───train_model.py      # Script per addestrare il modello
│   ├───predict_webcam.py   # Script per testare il modello via webcam (CLI)
│   ├───features.py         # Funzioni per estrazione e manipolazione feature
│   └───...
└───templates/
    └───index.html          # Pagina HTML dell'interfaccia web
```

### File e Cartelle Chiave

- **`app.py`**: Cuore dell'applicazione. È un server Flask che gestisce:
    - La pagina web principale (`index.html`).
    - L'endpoint `/predict` che riceve i frame video dal browser, esegue la predizione tramite il modello e restituisce il risultato in formato JSON.

- **`templates/index.html`**: L'interfaccia utente. Contiene il codice HTML e JavaScript per:
    - Accedere alla webcam.
    - Inviare i frame video al backend a intervalli regolari.
    - Ricevere le predizioni e visualizzarle.
    - Disegnare i landmarks della mano sopra il video.

- **`src/train_model.py`**: Script per l'addestramento del modello. Legge il file `lis_landmarks.csv`, esegue l'augmentation dei dati (creando varianti specchiate e ruotate) e addestra un `RandomForestClassifier`. Infine, salva il modello addestrato in `models/lis_model.pkl`.

- **`src/predict_webcam.py`**: Un'utilità da riga di comando per testare il modello in tempo reale senza avviare l'applicazione web. Utile per il debug.

- **`models/lis_model.pkl`**: Il modello di classificazione serializzato (salvato) con Joblib.

- **`data/processed/lis_landmarks.csv`**: Il dataset. Ogni riga rappresenta un gesto, con le coordinate dei landmarks della mano come feature e una colonna `target` con l'etichetta del gesto.

---

## 3. Istruzioni di Avvio

Per eseguire il progetto in locale, seguire questi passaggi.

### Prerequisiti

- Python 3.8+
- `pip` per la gestione dei pacchetti

### Installazione Dipendenze

Aprire un terminale nella cartella principale del progetto ed eseguire:

```bash
pip install -r requirements.txt
```

### Avvio dell'Applicazione Web

Per avviare il server Flask e l'interfaccia web:

```bash
python app.py
```

Successivamente, aprire il browser all'indirizzo [http://localhost:5000](http://localhost:5000).

### Eseguire l'Addestramento

Per ri-addestrare il modello con dati aggiornati:

```bash
python src/train_model.py
```

Questo sovrascriverà il file `models/lis_model.pkl` con la nuova versione.

---

## 4. Dipendenze

Le librerie Python necessarie sono elencate in `requirements.txt`:

- **`flask`**: Per il server web.
- **`opencv-python`**: Per l'elaborazione delle immagini.
- **`mediapipe`**: Per il rilevamento dei landmarks della mano.
- **`pandas`**: Per la manipolazione dei dati.
- **`numpy`**: Per calcoli numerici.
- **`scikit-learn`**: Per l'addestramento e l'uso del modello di classificazione.
- **`joblib`**: Per salvare e caricare il modello.

---

## 5. Future Evoluzioni

Questo file può essere usato come base per aggiungere nuove sezioni e migliorare il progetto. Alcune idee per future sezioni o modifiche:

- **Aggiungere nuovi gesti:**
    1.  **Raccolta Dati:** Registrare nuovi video dei gesti e usare uno script (da creare) per estrarre i landmarks e aggiungerli a `lis_landmarks.csv`.
    2.  **Addestramento:** Eseguire `python src/train_model.py` per creare un nuovo modello che includa i nuovi gesti.

- **Migliorare il Modello:**
    - Sperimentare con altri algoritmi di classificazione (es. `XGBoost`, `SVM`).
    - Ottimizzare gli iperparametri del `RandomForestClassifier` attuale.
    - Introdurre feature temporali per riconoscere gesti dinamici (che richiedono una sequenza di movimenti).

- **Migliorare l'Interfaccia:**
    - Aggiungere una cronologia dei segni riconosciuti.
    - Mostrare il livello di confidenza della predizione.
    - Creare una pagina di "training" dove l'utente può registrare nuovi gesti direttamente dal browser.
