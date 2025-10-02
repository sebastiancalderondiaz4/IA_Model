import cv2, time
import numpy as np
import tflite_runtime.interpreter as tfi
from main import predict_img_bytes  # Asumiendo que el archivo `main.py` tiene la función `predecir_imagen`

# Inicializa el modelo (debe estar configurado para no cargar de nuevo el modelo cada vez)
model_path = './modelo_mobilenetv2_entrenado.tflite'
interpreter = tfi.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Captura de video desde la cámara
cap = cv2.VideoCapture(0)  # 0 es el ID de la cámara predeterminada

if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error al capturar imagen.")
        break

    # Redimensiona la imagen y normaliza
    frame_resized = cv2.resize(frame, (224, 224))
    frame_normalized = frame_resized.astype('float32') / 255.0
    frame_normalized = np.expand_dims(frame_normalized, axis=0)

    # Predicción
    prediccion = predict_img_bytes(frame_normalized, interpreter)
    
#    time.sleep(2)
    # Muestra el resultado
    cv2.putText(frame, f"Prediccion: {prediccion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Muestra la imagen
    cv2.imshow('Camara en vivo', frame)

    # Si presionas 'q', salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la cámara y cierra las ventanas
cap.release()
cv2.destroyAllWindows()
