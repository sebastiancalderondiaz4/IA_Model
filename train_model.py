import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ==============================================================================
# 1. DEFINICIÓN DE PATHS Y ESTRUCTURA DEL DATASET
# ==============================================================================

# **IMPORTANTE:** Reemplaza esta ruta con la ubicación real de tu carpeta 'garbage_classification' en tu sistema local.
PATH_DATA = "C:/Users/User/Desktop/garbage_classification"

# Ejemplo: PATH_DATA = "D:\\MisProyectos\\garbage_dataset\\garbage_classification\\"


def cargar_imagenes(dataset_path, map_categories, img_size=(224, 224)):
    """Carga, redimensiona y normaliza las imágenes del dataset."""
    categorias = map_categories.keys()
    imagenes = []
    etiquetas = []

    # Cargar imágenes por categoría
    for idx, categoria in enumerate(categorias):
        for garbage_type in map_categories[categoria]:
            # Contruyo la ruta para llegar al path donde se almacenan las imagenes
            categoria_path = os.path.join(dataset_path, categoria, garbage_type)
            
            # Verificar si la ruta existe para evitar errores
            if not os.path.exists(categoria_path):
                print(f"Advertencia: Ruta no encontrada: {categoria_path}")
                continue

            # Recorro las imagenes
            for archivo in os.listdir(categoria_path):
                if archivo.lower().endswith(('.jpg', '.png', '.webp')):
                    img_path = os.path.join(categoria_path, archivo)
                    img = cv2.imread(img_path)
                    
                    if img is None:
                        print(f"Error al cargar la imagen: {img_path}")
                        continue
                        
                    img = cv2.resize(img, img_size)  # Redimensionar a 224x224
                    img = img.astype('float32') / 255.0  # Normalizar a [0, 1]
                    
                    imagenes.append(img)
                    etiquetas.append(idx)  # Etiquetar con el índice de la categoría

    return np.array(imagenes), np.array(etiquetas)


# Definir la estructura del dataset a partir de los subdirectorios
try:
    map_categories = {
        "APROVECHABLES" : os.listdir(os.path.join(PATH_DATA, "APROVECHABLES")),
        "NO_APROVECHABLE" : os.listdir(os.path.join(PATH_DATA, "NO_APROVECHABLE")),
        "ORGANICO" : os.listdir(os.path.join(PATH_DATA, "ORGANICO"))
    }
except FileNotFoundError as e:
    print(f"ERROR: La ruta base del dataset es incorrecta o faltan carpetas: {e}")
    print(f"Asegúrate de que PATH_DATA ({PATH_DATA}) apunte a la carpeta contenedora.")
    exit()
    

# ==============================================================================
# 2. CARGA DE DATOS Y PREPROCESAMIENTO
# ==============================================================================

print("Cargando y preprocesando imágenes...")
imagenes, etiquetas = cargar_imagenes(PATH_DATA, map_categories)

if len(imagenes) == 0:
    print("No se encontraron imágenes válidas. Terminando el script.")
    exit()

# Dividir en entrenamiento y validación
print(f"Total de imágenes cargadas: {len(imagenes)}")
X_train, X_val, y_train, y_val = train_test_split(
    imagenes, etiquetas, test_size=0.2, stratify=etiquetas, random_state=42
)

# Convertir las etiquetas a formato one-hot (3 clases)
num_classes = len(map_categories.keys())
y_train = to_categorical(y_train, num_classes=num_classes)
y_val = to_categorical(y_val, num_classes=num_classes)


# ==============================================================================
# 3. CREACIÓN Y ENTRENAMIENTO DEL MODELO
# ==============================================================================

def crear_modelo():
    """Crea y compila el modelo MobileNetV2 para Transfer Learning."""
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Congelar las capas del modelo base
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),  # Capa de pooling global
        Dropout(0.2),              # Para evitar sobreajuste
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')  # Tres categorías
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

modelo = crear_modelo()
print("\nResumen del Modelo MobileNetV2:")
modelo.summary()

# Usar early stopping para evitar sobreajuste
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entrenar el modelo
print("\nIniciando el entrenamiento...")
history = modelo.fit(
    X_train, y_train, 
    epochs=30, 
    batch_size=32, 
    validation_data=(X_val, y_val), 
    callbacks=[early_stopping]
)


# ==============================================================================
# 4. EVALUACIÓN Y PREDICCIÓN
# ==============================================================================

# Evaluación del modelo en bajo nivel
pérdida, precisión = modelo.evaluate(X_val, y_val, verbose=0)
print(f"\nPrecisión en Validación: {precisión*100:.2f}%")
print(f"Pérdida en Validación: {pérdida:.4f}")

# Hacemos predicciones para la Matriz de Confusión
y_val_indices = np.argmax(y_val, axis=1)
y_pred = modelo.predict(X_val)
y_pred_indices = np.argmax(y_pred, axis=1)

# Crear matriz de confusión
clases = list(map_categories.keys())
matriz_confusion = confusion_matrix(y_val_indices, y_pred_indices)

# Mostrar la matriz de confusión
disp = ConfusionMatrixDisplay(confusion_matrix=matriz_confusion, display_labels=clases)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Matriz de Confusión")
plt.show()

def predecir_imagen(imagen_path):
    """Carga, preprocesa y predice la categoría de una sola imagen."""
    if not os.path.exists(imagen_path):
        print(f"Error: La imagen de prueba no se encontró en la ruta: {imagen_path}")
        return

    # cv2.IMREAD_COLOR = 1 (Asegura la carga a color)
    img = cv2.imread(imagen_path, 1) 
    
    if img is None:
        print(f"Error: No se pudo leer la imagen en la ruta: {imagen_path}")
        return
        
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # Expande la dimensión para el batch

    prediccion = modelo.predict(img)
    categoria_idx = np.argmax(prediccion, axis=1)
    
    print(f"\n--- Resultado de Predicción ---")
    print(f"Predicción: {clases[categoria_idx[0]]}")
    print(f"Probabilidades: {dict(zip(clases, prediccion[0]))}")
    
# Ruta de prueba (AJUSTAR PARA EL ENTORNO LOCAL)
PATH_IMAGE_TEST = os.path.join(PATH_DATA, "APROVECHABLES", "papel", "cartyon.webp") 

# Ejemplo de predicción
predecir_imagen(PATH_IMAGE_TEST)


# ==============================================================================
# 5. GUARDAR EL MODELO
# ==============================================================================
# Guardar en el directorio local del proyecto
MODEL_SAVE_PATH = 'modelo_mobilenetv2_entrenado.keras'
modelo.save(MODEL_SAVE_PATH)
print(f"\nModelo guardado exitosamente en: {MODEL_SAVE_PATH}")