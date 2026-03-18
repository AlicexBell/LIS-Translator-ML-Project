import joblib
import numpy as np
import os

# Carichiamo il modello e l'estrattore
def get_prediction(landmarks, model_path="./models/lis_model.pkl"):
    if not os.path.exists(model_path):
        return "Modello non trovato"
    
    model = joblib.load(model_path)
    
    # I landmarks arrivano come lista [x0, y0, z0, x1, y1...]
    # Dobbiamo assicurarci che siano normalizzati (sottraendo il polso)
    # se non lo sono già stati fatti nel ponte JS/Python
    
    prediction = model.predict([landmarks])
    return prediction[0]