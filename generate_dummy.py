
import os
import numpy as np
from PIL import Image

def create_dummy_data():
    base_dir = "dataset_isic_mini/train"
    classes = ["Melanoma", "Nevus"]
    
    print("Generando datos sintéticos para prueba de entrenamiento...")
    
    for cls in classes:
        cls_dir = os.path.join(base_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        
        # Crear 20 imágenes dummy por clase
        for i in range(20):
            # Melanoma: Ruido rojizo/oscuro, Nevus: Ruido rosado/claro
            color = (random.randint(50, 150), 0, 0) if cls == "Melanoma" else (200, 150, 150)
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            # Tintar un poco para diferenciar
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] + color[0], 0, 255)
            
            img = Image.fromarray(img_array)
            img.save(os.path.join(cls_dir, f"{cls}_{i}.jpg"))
            
    print("Datos sintéticos creados. Listo para probar 'train_model.py'.")

import random
if __name__ == "__main__":
    create_dummy_data()
