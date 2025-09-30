import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model_path = './modelo_mobilenetv2_entrenado.keras'

map_categories = {
    "aprovechable" : ['cardboard', 'paper', 'plastic'],
    "inorganica" : ['trash', 'white-glass'],
    "organicos" : ['biological']
}

def predecir_imagen(imagen_path, modelo):
    img = cv2.imread(imagen_path) # <- Al integrar la camara me da directamente el objeto img, no necesito leer de un fichero
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # Expande la dimensión para que sea compatible con el modelo

    prediccion = modelo.predict(img)
    categoria_idx = np.argmax(prediccion, axis=1)
    categorias = list(map_categories.keys())


    print(f"Predicción: {categorias[categoria_idx[0]]}")

modelo = load_model(model_path)


print(predecir_imagen('./OIP.jpg', modelo))