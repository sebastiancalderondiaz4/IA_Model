import os
import cv2
import numpy as np
#from tensorflow.keras.models import load_model
import tflite_runtime.interpreter  as tfi

model_path = './modelo_mobilenetv2_entrenado.tflite'

interpreter = tfi.Interpreter(model_path=model_path)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


map_categories = {
    "APROVECHABLES" : ['cardboard', 'paper', 'plastic'],
    "NO_APROVECHABLE" : ['trash', 'white-glass','trash2' ],
    "ORGANICOS" : ['biological','biological2','biological3' ]

}

def predecir_imagen(imagen_path, modelo):
    img = cv2.imread(imagen_path) # <- Al integrar la camara me da directamente el objeto img, no necesito leer de un fichero
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # Expande la dimensión para que sea compatible con el modelo

    modelo.set_tensor(input_details[0]['index'], img)

    modelo.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    prediction = np.argmax(prediction)

    prediction = list(map_categories.keys())[prediction]

    return prediction

def predict_img_bytes(imagen_bytes, modelo):
#    img = cv2.imread(imagen_path) # <- Al integrar la camara me da directamente el objeto img, no necesito leer de un fichero
#    img = cv2.resize(img, (224, 224))
#    img = img.astype('float32') / 255.0
#    img = np.expand_dims(img, axis=0)  # Expande la dimensión para que sea compatible con el modelo
    modelo.set_tensor(input_details[0]['index'], imagen_bytes)

    modelo.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    prediction = np.argmax(prediction)

    prediction = list(map_categories.keys())[prediction]

    return prediction


#prediccion = modelo.predict(img)
#    categoria_idx = np.argmax(prediccion, axis=1)
#    categorias = list(map_categories.keys())


    #print(f"Predicción: {categorias[categoria_idx[0]]}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Uso: python3 main.py <path_image>")
        sys.exit(1)

    img_path = sys.argv[1]
    categoria = predecir_imagen(img_path, interpreter)
    print(categoria)
