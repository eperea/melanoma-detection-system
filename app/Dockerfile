# Usar imagen oficial de TensorFlow que ya incluye TF 2.15
FROM tensorflow/tensorflow:2.15.0

WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Instalar PyTorch versión CPU (mucho más ligero: ~150MB vs 1GB)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copiar requirements
COPY requirements.txt .

# Instalar el resto de dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py"]
