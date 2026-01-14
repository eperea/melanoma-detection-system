# Usar imagen oficial de TensorFlow que ya incluye TF 2.15
FROM tensorflow/tensorflow:2.15.0

WORKDIR /app

# Instalar dependencias del sistema necesarias para OpenCV y otros
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias de Python (sin TF ni Keras, ya incluidos)
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Exponer el puerto de Streamlit
EXPOSE 8501

# Comando para ejecutar la aplicación
CMD ["streamlit", "run", "main.py"]
