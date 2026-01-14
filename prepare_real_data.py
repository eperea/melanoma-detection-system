
import os
import requests
import zipfile
import io
import shutil
from pathlib import Path

# URL de un dataset educativo con imágenes reales de ISIC (Melanoma vs Nevus)
# Este repositorio contiene un subconjunto curado para pruebas rápidas
DATASET_URL = "https://github.com/laxmimerit/Skin-Cancer-Detection-using-Machine-Learning/archive/refs/heads/master.zip"

DATASET_DIR = "dataset_isic_mini"

def download_real_data():
    print("Descargando imágenes REALES de dermatoscopía (esto puede tardar unos minutos)...")
    
    try:
        r = requests.get(DATASET_URL)
        if r.status_code == 200:
            print("Descarga completada. Descomprimiendo...")
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall("temp_download")
            
            # El zip tiene una estructura específica, vamos a mover las imágenes
            # a nuestra estructura limpia train/Melanoma y train/Nevus
            
            # Origen (depende del repo)
            base_extract = os.path.join("temp_download", "Skin-Cancer-Detection-using-Machine-Learning-master", "skin_cancer_dataset")
            
            # Destino
            target_train = os.path.join(DATASET_DIR, "train")
            
            if os.path.exists(base_extract):
                print("Organizando imágenes en carpetas de entrenamiento...")
                
                # Copiar carpetas
                # El repo suele tener 'benign' y 'malignant'. Mapeamos a Nevus y Melanoma.
                
                # Mapeo: Benign -> Nevus
                src_benign = os.path.join(base_extract, "train", "benign")
                dst_nevus = os.path.join(target_train, "Nevus")
                if os.path.exists(src_benign):
                    copy_images(src_benign, dst_nevus)
                
                # Mapeo: Malignant -> Melanoma
                src_malignant = os.path.join(base_extract, "train", "malignant")
                dst_melanoma = os.path.join(target_train, "Melanoma")
                if os.path.exists(src_malignant):
                    copy_images(src_malignant, dst_melanoma)
                    
                print("¡Dataset real preparado exitosamente!")
                
                # Limpieza
                shutil.rmtree("temp_download")
            else:
                print("La estructura del zip descargado no es la esperada. Revisa 'temp_download'.")
        else:
            print(f"Error descargando: Status {r.status_code}")
            
    except Exception as e:
        print(f"Error crítico: {e}")

def copy_images(src, dst):
    os.makedirs(dst, exist_ok=True)
    files = os.listdir(src)
    print(f"Copiando {len(files)} imágenes de {os.path.basename(src)} a {os.path.basename(dst)}...")
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))

if __name__ == "__main__":
    download_real_data()
