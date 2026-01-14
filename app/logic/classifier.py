import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image, ImageOps

class MelanomaClassifier:
    def __init__(self, model_path, labels_path):
        """
        Carga el modelo de clasificación Melanoma/Nevus entrenado con MobileNetV2.
        """
        self.model = keras.models.load_model(model_path, compile=False)
        
        # Cargar etiquetas correctamente
        # El archivo puede tener formato "0 Melanoma" o solo "Melanoma"
        self.labels = []
        with open(labels_path, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line:  # Ignorar líneas vacías
                    # Si la línea tiene formato "0 Melanoma", extraer solo el nombre
                    parts = line.split(" ", 1)
                    if len(parts) == 2 and parts[0].isdigit():
                        self.labels.append(parts[1])  # Solo tomar "Melanoma"
                    else:
                        self.labels.append(line)  # Tomar la línea completa
        
        print(f"[Classifier] Etiquetas cargadas: {self.labels}")
            
    def predict(self, image_array):
        """
        Clasifica una imagen de lesión cutánea.
        
        Args:
            image_array: numpy array (H, W, 3) uint8 RGB
            
        Returns:
            class_name: Nombre de la clase predicha ("Melanoma" o "Nevus")
            confidence_score: Confianza de la predicción (0-1)
            probabilities: Array con probabilidades de cada clase
        """
        target_size = (224, 224)
        
        # Convertir numpy array a PIL Image
        image = Image.fromarray(image_array)
        
        # Redimensionar manteniendo aspecto y recortando al centro
        image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
        
        # Convertir a array y normalizar (IGUAL que en entrenamiento: rescale=1./255)
        image_array_processed = np.asarray(image)
        normalized_image_array = image_array_processed.astype(np.float32) / 255.0
        
        # Preparar batch de 1 imagen
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        
        # Predecir
        prediction = self.model.predict(data, verbose=0)
        index = np.argmax(prediction)
        class_name = self.labels[index] if index < len(self.labels) else f"Clase {index}"
        confidence_score = float(prediction[0][index])
        
        return class_name, confidence_score, prediction[0]
