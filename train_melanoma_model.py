"""
Script de Entrenamiento - Clasificador Melanoma vs Nevus
=========================================================
Este script entrena un modelo de clasificación binaria usando Transfer Learning
con MobileNetV2 para distinguir entre Melanoma (maligno) y Nevus (benigno).

Dataset: HAM10000 (base_dir)
Optimizado para múltiples CPUs
"""

import os
import sys

# ============================================
# CONFIGURACIÓN DE TENSORFLOW PARA MULTI-CPU
# ============================================
# Configurar ANTES de importar TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Reducir logs

import tensorflow as tf

# Configurar para usar 4 CPUs (hilos)
NUM_CPUS = 4
tf.config.threading.set_intra_op_parallelism_threads(NUM_CPUS)
tf.config.threading.set_inter_op_parallelism_threads(NUM_CPUS)

print(f"🖥️  TensorFlow {tf.__version__} configurado para usar {NUM_CPUS} CPUs")

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np

# ============================================
# CONFIGURACIÓN DEL ENTRENAMIENTO
# ============================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32  # Optimizado para CPU con suficiente RAM
EPOCHS = 15      # Épocas totales (puede detenerse antes con EarlyStopping)
LEARNING_RATE = 0.0001

# Rutas - Usar la carpeta base_dir con imágenes reales
BASE_DATA_DIR = os.path.join("..", "base_dir", "base_dir")
TRAIN_DIR = os.path.join(BASE_DATA_DIR, "train_dir")
VAL_DIR = os.path.join(BASE_DATA_DIR, "val_dir")

# Rutas de salida
OUTPUT_MODEL_PATH = os.path.join("app", "models", "keras_model.h5")  # Sobrescribir el modelo existente
OUTPUT_LABELS_PATH = os.path.join("app", "models", "labels.txt")

# Workers para carga de datos en paralelo
NUM_WORKERS = 4


def train_model():
    print("=" * 60)
    print("🔬 ENTRENAMIENTO - Clasificador Melanoma vs Nevus")
    print("=" * 60)
    print(f"📅 Fecha: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Verificar que existan las carpetas de datos
    train_mel = os.path.join(TRAIN_DIR, "mel")
    train_nv = os.path.join(TRAIN_DIR, "nv")
    val_mel = os.path.join(VAL_DIR, "mel")
    val_nv = os.path.join(VAL_DIR, "nv")
    
    if not os.path.exists(train_mel) or not os.path.exists(train_nv):
        print(f"❌ Error: No se encontraron las carpetas de entrenamiento:")
        print(f"   - {train_mel}")
        print(f"   - {train_nv}")
        return False
    
    # Contar imágenes disponibles
    mel_train_count = len([f for f in os.listdir(train_mel) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    nv_train_count = len([f for f in os.listdir(train_nv) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    mel_val_count = len([f for f in os.listdir(val_mel) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(val_mel) else 0
    nv_val_count = len([f for f in os.listdir(val_nv) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(val_nv) else 0
    
    total_train = mel_train_count + nv_train_count
    total_val = mel_val_count + nv_val_count
    
    print(f"\n📊 Dataset HAM10000:")
    print(f"   ┌─ Entrenamiento ─────────────────")
    print(f"   │  Melanoma (mel):  {mel_train_count:,} imágenes")
    print(f"   │  Nevus (nv):      {nv_train_count:,} imágenes")
    print(f"   │  Total:           {total_train:,} imágenes")
    print(f"   ├─ Validación ─────────────────────")
    print(f"   │  Melanoma (mel):  {mel_val_count:,} imágenes")
    print(f"   │  Nevus (nv):      {nv_val_count:,} imágenes")
    print(f"   │  Total:           {total_val:,} imágenes")
    print(f"   └─────────────────────────────────")
    
    # ============================================
    # DATA AUGMENTATION
    # ============================================
    print("\n🔄 Configurando Data Augmentation...")
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # ============================================
    # CARGAR DATOS
    # ============================================
    print("📂 Cargando generadores de datos...")
    
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=['mel', 'nv'],  # Solo Melanoma y Nevus
        shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=['mel', 'nv'],
        shuffle=False
    )
    
    # Mapeo de clases
    class_indices = train_generator.class_indices
    print(f"   ✓ Clases: {class_indices}")
    
    # ============================================
    # CONSTRUIR MODELO
    # ============================================
    print("\n🧠 Construyendo modelo MobileNetV2...")
    
    # Modelo base preentrenado
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Congelar modelo base inicialmente
    base_model.trainable = False
    
    # Cabezal de clasificación personalizado
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compilar
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_weights])
    total_params = model.count_params()
    
    print(f"   ✓ Parámetros totales:      {total_params:,}")
    print(f"   ✓ Parámetros entrenables:  {trainable_params:,}")
    
    # ============================================
    # CALLBACKS
    # ============================================
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        ),
        ModelCheckpoint(
            OUTPUT_MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # ============================================
    # FASE 1: ENTRENAR CABEZAL
    # ============================================
    print("\n" + "=" * 60)
    print("📈 FASE 1: Entrenamiento del cabezal (base congelada)")
    print("=" * 60)
    
    steps_per_epoch = train_generator.samples // BATCH_SIZE
    validation_steps = max(1, validation_generator.samples // BATCH_SIZE)
    
    print(f"   Steps por época (train): {steps_per_epoch}")
    print(f"   Steps por época (val):   {validation_steps}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Workers: {NUM_WORKERS}")
    print("")
    
    history1 = model.fit(
        train_generator,
        epochs=7,
        validation_data=validation_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        workers=NUM_WORKERS,
        use_multiprocessing=False,  # Más estable en Windows
        verbose=1
    )
    
    # ============================================
    # FASE 2: FINE-TUNING
    # ============================================
    print("\n" + "=" * 60)
    print("📈 FASE 2: Fine-tuning (descongelando últimas capas)")
    print("=" * 60)
    
    # Descongelar últimas 30 capas
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Recompilar con learning rate menor
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    trainable_params_ft = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_weights])
    print(f"   ✓ Parámetros entrenables (fine-tune): {trainable_params_ft:,}")
    print("")
    
    epochs_done = len(history1.history['loss'])
    
    history2 = model.fit(
        train_generator,
        epochs=EPOCHS,
        initial_epoch=epochs_done,
        validation_data=validation_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        workers=NUM_WORKERS,
        use_multiprocessing=False,
        verbose=1
    )
    
    # ============================================
    # GUARDAR MODELO Y LABELS
    # ============================================
    print("\n💾 Guardando modelo y etiquetas...")
    
    os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)
    
    # Guardar modelo
    model.save(OUTPUT_MODEL_PATH)
    print(f"   ✓ Modelo: {OUTPUT_MODEL_PATH}")
    
    # Guardar labels
    name_map = {'mel': 'Melanoma', 'nv': 'Nevus'}
    with open(OUTPUT_LABELS_PATH, "w") as f:
        for class_name, idx in sorted(class_indices.items(), key=lambda x: x[1]):
            readable_name = name_map.get(class_name, class_name)
            f.write(f"{idx} {readable_name}\n")
    
    print(f"   ✓ Labels: {OUTPUT_LABELS_PATH}")
    
    # ============================================
    # EVALUACIÓN FINAL
    # ============================================
    print("\n" + "=" * 60)
    print("📊 EVALUACIÓN FINAL")
    print("=" * 60)
    
    val_loss, val_accuracy = model.evaluate(validation_generator, verbose=0)
    
    print(f"   Loss de validación:     {val_loss:.4f}")
    print(f"   Accuracy de validación: {val_accuracy:.2%}")
    
    # Mostrar métricas finales
    print("\n" + "=" * 60)
    print("✅ ¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
    print("=" * 60)
    print(f"\n🎯 El modelo está listo para detectar Melanoma vs Nevus")
    print(f"   Accuracy: {val_accuracy:.2%}")
    print(f"\n📁 Archivos generados:")
    print(f"   - {OUTPUT_MODEL_PATH}")
    print(f"   - {OUTPUT_LABELS_PATH}")
    print(f"\n🚀 Puedes ejecutar la aplicación con: streamlit run app/main.py")
    
    return True


if __name__ == "__main__":
    success = train_model()
    sys.exit(0 if success else 1)
