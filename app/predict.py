import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image

def preprocess_image(img_path, target_size=(224, 224)):
    """
    Preprocesa la imagen para que sea compatible con el modelo
    """
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalización
    
    return img_array

def predict_covid(model, img_array):
    """
    Realiza la predicción usando el modelo cargado
    """
    prediction = model.predict(img_array)
    return {
        'prediction': float(prediction[0][0]),
        'class': 'COVID-19' if prediction[0][0] > 0.5 else 'No COVID-19',
        'confidence': float(prediction[0][0]) if prediction[0][0] > 0.5 else float(1 - prediction[0][0])
    }