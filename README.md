# 🔬 Sistema de Detección de Melanoma con Super-Resolución

Sistema de apoyo al diagnóstico médico (CAD) para la detección temprana de melanoma utilizando técnicas de Deep Learning y mejora de imagen mediante Super-Resolución.

## 📋 Características

- **Super-Resolución (SRCNN)**: Mejora la calidad de las imágenes dermatoscópicas antes del análisis
- **Clasificación con Deep Learning**: Modelo MobileNetV2 entrenado con el dataset HAM10000
- **Interfaz Web Intuitiva**: Aplicación Streamlit para uso médico
- **Gestión de Pacientes**: Base de datos PostgreSQL para historial médico
- **Reportes PDF**: Generación automática de informes descargables

## 🧠 Tecnologías

| Componente | Tecnología |
|------------|------------|
| Super-Resolución | PyTorch (SRCNN) |
| Clasificación | TensorFlow/Keras (MobileNetV2) |
| Interfaz Web | Streamlit |
| Base de Datos | PostgreSQL |
| Contenedores | Docker |

## 🚀 Instalación

### Requisitos Previos
- Python 3.11+
- Docker (opcional, para PostgreSQL)

### Pasos

1. Clonar el repositorio:
```bash
git clone https://github.com/eperea/melanoma-detection-system.git
cd melanoma-detection-system
```

2. Crear entorno virtual e instalar dependencias:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r app/requirements.txt
```

3. (Opcional) Iniciar PostgreSQL con Docker:
```bash
docker-compose up -d db
```

4. Ejecutar la aplicación:
```bash
cd app
streamlit run main.py
```

5. Abrir en el navegador: http://localhost:8501

## 📊 Rendimiento del Modelo

| Métrica | Valor |
|---------|-------|
| Accuracy (Validación) | 96.7% |
| F1-Score (Melanoma) | 0.93 |
| F1-Score (Nevus) | 0.94 |

## 📁 Estructura del Proyecto

```
Proyecto_SR_Final/
├── app/
│   ├── main.py              # Aplicación Streamlit
│   ├── database.py          # Conexión PostgreSQL
│   ├── logic/
│   │   ├── classifier.py    # Clasificador Melanoma/Nevus
│   │   └── sr_model.py      # Modelo Super-Resolución
│   ├── models/
│   │   ├── keras_model.h5   # Modelo de clasificación
│   │   ├── best_srcnn.pth   # Modelo SR
│   │   └── labels.txt       # Etiquetas
│   └── utils/
│       └── pdf_report.py    # Generador de PDF
├── train_melanoma_model.py  # Script de entrenamiento
├── docker-compose.yml       # Configuración Docker
└── README.md
```

## ⚠️ Aviso Legal

Este sistema es una **herramienta de apoyo diagnóstico** y no sustituye la evaluación de un profesional médico especializado. Los resultados deben ser siempre validados por un dermatólogo.

## 👨‍💻 Autor

Proyecto de Tesis - Detección de Melanoma mediante IA

---
*Desarrollado con Python, TensorFlow, PyTorch y Streamlit*
