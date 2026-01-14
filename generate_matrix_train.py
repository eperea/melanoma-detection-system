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

# ⚠️ AHORA USAMOS EL DATASET DE ENTRENAMIENTO PARA VERIFICAR
EVAL_DIR = "../base_dir/base_dir/train_dir" 

def plot_confusion_matrix():
    print("🔄 Cargando modelo...")
    try:
        model = load_model(MODEL_PATH)
        print("✅ Modelo cargado correctamente.")
    except Exception as e:
        print(f"❌ Error cargando el modelo: {e}")
        return

    print(f"📂 Preparando datos de ENTRENAMIENTO desde: {EVAL_DIR}")
    # Usamos ImageDataGenerator sin Data Augmentation para evaluación pura
    eval_datagen = ImageDataGenerator(rescale=1./255)

    eval_generator = eval_datagen.flow_from_directory(
        EVAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=['mel', 'nv'],  # Solo evaluamos Melanoma y Nevus
        shuffle=False # CRÍTICO: No barajar para comparar con etiquetas reales
    )

    # Nombres de las clases
    class_names = list(eval_generator.class_indices.keys()) 
    display_names = ['Melanoma', 'Nevus'] 

    print(f"✅ Se encontraron {eval_generator.samples} imágenes para evaluación.")

    # Realizar predicciones
    print("🔮 Generando predicciones masivas (12,000 imágenes)... por favor espera...")
    predictions = model.predict(eval_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Obtener etiquetas reales
    true_classes = eval_generator.classes
    
    # Generar Matriz de Confusión
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # ---- 1. Generar Reporte de Texto ----
    print("\n" + "="*50)
    print("📊 REPORTE DE CLASIFICACIÓN (ENTRENAMIENTO)")
    print("="*50)
    report = classification_report(true_classes, predicted_classes, target_names=display_names)
    print(report)
    
    with open("reporte_train.txt", "w") as f:
        f.write(report)

    # ---- 2. Generar Gráfico Visual ----
    plt.figure(figsize=(10, 8))
    
    # Mapa de calor (Heatmap)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', # Usamos verde para diferenciar
                xticklabels=display_names,
                yticklabels=display_names,
                annot_kws={"size": 14}) # Números más grandes
    
    plt.title('Matriz de Confusión: Dataset de Entrenamiento', fontsize=16)
    plt.ylabel('Etiqueta Real', fontsize=12)
    plt.xlabel('Predicción del Modelo', fontsize=12)
    
    # Guardar imagen
    output_image = "matriz_confusion_train.png"
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"\n💾 Gráfico guardado como: {output_image}")
    print("\n✅ ¡Proceso completado!")

if __name__ == "__main__":
    plot_confusion_matrix()
