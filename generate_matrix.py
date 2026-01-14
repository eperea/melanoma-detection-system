import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_PATH = "app/models/keras_model.h5"
VAL_DIR = "../base_dir/base_dir/val_dir"

def plot_confusion_matrix():
    print("🔄 Cargando modelo...")
    try:
        model = load_model(MODEL_PATH)
        print("✅ Modelo cargado correctamente.")
    except Exception as e:
        print(f"❌ Error cargando el modelo: {e}")
        return

    print("📂 Preparando datos de validación...")
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Es CRÍTICO usar shuffle=False para que las predicciones coincidan con las etiquetas reales
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=['mel', 'nv'],  # Solo evaluamos Melanoma y Nevus
        shuffle=False
    )

    # Nombres de las clases
    class_names = list(val_generator.class_indices.keys()) # ['mel', 'nv']
    display_names = ['Melanoma', 'Nevus'] # Nombres bonitos para la gráfica

    print(f"✅ Se encontraron {val_generator.samples} imágenes para validación.")

    # Realizar predicciones
    print("🔮 Generando predicciones (esto puede tardar unos segundos)...")
    predictions = model.predict(val_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Obtener etiquetas reales
    true_classes = val_generator.classes
    
    # Generar Matriz de Confusión
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # ---- 1. Generar Reporte de Texto ----
    print("\n" + "="*50)
    print("📊 REPORTE DE CLASIFICACIÓN")
    print("="*50)
    report = classification_report(true_classes, predicted_classes, target_names=display_names)
    print(report)
    
    # Guardar reporte en texto
    with open("reporte_clasificacion.txt", "w") as f:
        f.write(report)

    # ---- 2. Generar Gráfico Visual ----
    plt.figure(figsize=(8, 6))
    
    # Mapa de calor (Heatmap)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=display_names,
                yticklabels=display_names)
    
    plt.title('Matriz de Confusión: Melanoma vs Nevus')
    plt.ylabel('Etiqueta Real (Verdad)')
    plt.xlabel('Predicción del Modelo')
    
    # Guardar imagen
    output_image = "matriz_confusion.png"
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"\n💾 Gráfico guardado como: {output_image}")
    print(f"💾 Reporte guardado como: reporte_clasificacion.txt")
    print("\n✅ ¡Proceso completado!")

if __name__ == "__main__":
    plot_confusion_matrix()
