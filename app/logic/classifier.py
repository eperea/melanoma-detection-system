import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image, ImageOps

# --- MONKEY PATCHING PARA ARREGLAR ERROR 'groups' ---
# El modelo MobileNetV2 guardado tiene 'groups=1' en su configuración,
# lo cual es inválido en versiones recientes de Keras/TensorFlow.
# Parcheamos la clase original directamente.

OriginalDepthwiseConv2D = keras.layers.DepthwiseConv2D

class CustomDepthwiseConv2D(OriginalDepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        # Eliminar 'groups' si existe en los argumentos
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

# Reemplazar la clase en el módulo keras.layers
keras.layers.DepthwiseConv2D = CustomDepthwiseConv2D
# También se necesita inyectar en los objetos personalizados por si acaso
# ----------------------------------------------------

class MelanomaClassifier:
    def __init__(self, model_path, labels_path):
        """
        Carga el modelo de clasificación Melanoma/Nevus entrenado con MobileNetV2.
        """
        # Load model explicitly passing context
        custom_objects = {'DepthwiseConv2D': CustomDepthwiseConv2D}
        
        try:
            print(f"[Classifier] Intentando cargar modelo desde {model_path}...")
            self.model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            print("[Classifier] ✅ Modelo cargado exitosamente.")
        except Exception as e:
            print(f"❌ Error crítico cargando modelo: {e}")
            # Intentar estrategia de fallback extrema: cargar configuración y pesos por separado si fuera posible
            # pero por ahora relanzamos para ver el error en logs
            raise e
        
        # Cargar etiquetas
        self.labels = []
        try:
            with open(labels_path, "r") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        parts = line.split(" ", 1)
                        if len(parts) == 2 and parts[0].isdigit():
                            self.labels.append(parts[1])
                        else:
                            self.labels.append(line)
        except Exception as e:
            print(f"❌ Error cargando etiquetas: {e}")
            self.labels = ["Melanoma", "Nevus"]
        
        print(f"[Classifier] Etiquetas cargadas: {self.labels}")
            
    def predict(self, image_array):
        # ... (código igual al anterior) ...
        target_size = (224, 224)
        image = Image.fromarray(image_array)
        image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
        image_array_processed = np.asarray(image)
        normalized_image_array = image_array_processed.astype(np.float32) / 255.0
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        
        prediction = self.model.predict(data, verbose=0)
        index = np.argmax(prediction)
        class_name = self.labels[index] if index < len(self.labels) else f"Clase {index}"
        confidence_score = float(prediction[0][index])
        
        return class_name, confidence_score, prediction[0]
