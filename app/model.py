from tensorflow.keras.models import load_model
import os

def load_covid_model(model_path):
    """
    Carga el modelo pre-entrenado de COVID-19
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    return load_model(model_path)