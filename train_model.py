
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import os

# Configuración
IMG_SIZE = (224, 224)
BATCH_SIZE = 16  # Pequeño para 16GB RAM y CPU
EPOCHS = 10      # Pocas épocas para prueba rápida en CPU
DATA_DIR = "dataset_isic_mini"

def train_model():
    print("Iniciando entrenamiento...")
    
    # 1. Data Augmentation (Clave para pocos datos)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    try:
        train_generator = train_datagen.flow_from_directory(
            os.path.join(DATA_DIR, "train"),
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training'
        )

        validation_generator = train_datagen.flow_from_directory(
            os.path.join(DATA_DIR, "train"),
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation'
        )
    except Exception as e:
        print(f"Error cargando datos: {e}")
        print(f"Asegúrate de que existan las carpetas '{DATA_DIR}/train/Melanoma' y '{DATA_DIR}/train/Nevus' con imágenes.")
        return

    # 2. Modelo Base (MobileNetV2 - Transfer Learning)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Congelar capas base (para no dañar lo que ya sabe)
    base_model.trainable = False

    # 3. Cabezal de Clasificación Personalizado
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x) # Evitar overfitting con pocos datos
    predictions = Dense(2, activation='softmax')(x) # 2 Clases: Melanoma, Nevus

    model = Model(inputs=base_model.input, outputs=predictions)

    # 4. Compilar
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("Modelo compilado. Entrenando...")

    # 5. Entrenar
    try:
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=validation_generator
        )
        
        # 6. Guardar
        model.save("app/models/keras_model_v2.h5")
        print("¡Entrenamiento completado! Modelo guardado en 'app/models/keras_model_v2.h5'")
        
        # Guardar labels
        with open("app/models/labels_v2.txt", "w") as f:
            # Obtener orden de clases del generador
            indices = train_generator.class_indices
            labels = {v: k for k, v in indices.items()}
            for i in range(len(labels)):
                f.write(f"{i} {labels[i]}\n")
                
    except Exception as e:
        print(f"Error durante entrenamiento: {e}")

if __name__ == "__main__":
    train_model()
