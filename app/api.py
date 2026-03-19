"""
API REST para integración con Raspberry Pi
Este módulo expone endpoints para recibir imágenes desde dispositivos externos
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os
import uuid
from datetime import datetime
import pytz

# Importar lógica de análisis
from logic.sr_model import SRPredictor
from logic.classifier import MelanomaClassifier
from database import init_database, registrar_paciente, guardar_analisis, buscar_paciente

# Configuración
app = FastAPI(
    title="Melanoma Detection API",
    description="API para integración con dispositivos Raspberry Pi",
    version="1.0.0"
)

# CORS para permitir conexiones desde Raspberry Pi
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rutas de modelos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SR_PATH = os.path.join(BASE_DIR, "models", "best_srcnn.pth")
MODEL_CL_PATH = os.path.join(BASE_DIR, "models", "keras_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "models", "labels.txt")

# Directorio para guardar imágenes recibidas
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads", "raspberry")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Cargar modelos al iniciar
sr_predictor = None
classifier = None

@app.on_event("startup")
async def startup_event():
    """Carga modelos al iniciar el servidor"""
    global sr_predictor, classifier
    try:
        sr_predictor = SRPredictor(MODEL_SR_PATH)
        classifier = MelanomaClassifier(MODEL_CL_PATH, LABELS_PATH)
        init_database()
        print("✅ Modelos cargados correctamente")
    except Exception as e:
        print(f"❌ Error cargando modelos: {e}")


@app.get("/")
async def root():
    """Endpoint de verificación"""
    return {
        "status": "online",
        "service": "Melanoma Detection API",
        "version": "1.0.0",
        "timestamp": datetime.now(pytz.timezone("America/Bogota")).isoformat()
    }


@app.get("/health")
async def health_check():
    """Verificar estado del servicio y modelos"""
    return {
        "status": "healthy",
        "models_loaded": sr_predictor is not None and classifier is not None,
        "timestamp": datetime.now(pytz.timezone("America/Bogota")).isoformat()
    }


@app.post("/api/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    device_id: str = Form(default="raspberry_001"),
    patient_id: str = Form(default=None),
    location: str = Form(default="No especificada"),
    notes: str = Form(default="")
):
    """
    Analizar imagen enviada desde Raspberry Pi
    
    - **image**: Imagen JPG/PNG de la lesión
    - **device_id**: Identificador del dispositivo Raspberry Pi
    - **patient_id**: ID del paciente (opcional)
    - **location**: Ubicación de la lesión
    - **notes**: Notas adicionales
    """
    if sr_predictor is None or classifier is None:
        raise HTTPException(status_code=503, detail="Modelos no disponibles")
    
    # Validar formato de imagen
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Formato no soportado. Use JPG o PNG")
    
    try:
        # Leer imagen
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Guardar imagen original
        timestamp = datetime.now(pytz.timezone("America/Bogota")).strftime("%Y%m%d_%H%M%S")
        image_filename = f"{device_id}_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
        image_path = os.path.join(UPLOAD_DIR, image_filename)
        pil_image.save(image_path, "JPEG", quality=95)
        
        # Aplicar super resolución
        img_sr = sr_predictor.enhance(pil_image)
        
        # Clasificar
        class_label, confidence, all_probs = classifier.predict(img_sr)
        
        # Determinar resultado
        if confidence < 0.70:
            diagnosis = "Otra afectación / No clasificable"
            risk_level = "REVISAR"
            led_color = "YELLOW"
        elif class_label.lower() == "melanoma":
            diagnosis = "Posible Melanoma"
            risk_level = "ALTO"
            led_color = "RED"
        else:
            diagnosis = "Nevus Benigno"
            risk_level = "BAJO"
            led_color = "GREEN"
        
        # Preparar respuesta
        result = {
            "success": True,
            "timestamp": datetime.now(pytz.timezone("America/Bogota")).isoformat(),
            "device_id": device_id,
            "image_saved": image_filename,
            "analysis": {
                "diagnosis": diagnosis,
                "class": class_label,
                "confidence": round(confidence * 100, 2),
                "risk_level": risk_level,
                "probabilities": {k: round(v * 100, 2) for k, v in all_probs.items()} if all_probs else {}
            },
            "led_signal": led_color,  # Para controlar LED en Raspberry Pi
            "recommendation": get_recommendation(risk_level)
        }
        
        # Guardar en base de datos si hay paciente
        if patient_id:
            try:
                guardar_analisis(
                    paciente_id=patient_id,
                    imagen=contents,
                    resultado=class_label,
                    confianza=confidence,
                    ubicacion=location,
                    notas=f"[Raspberry:{device_id}] {notes}"
                )
                result["saved_to_db"] = True
            except Exception as db_error:
                result["saved_to_db"] = False
                result["db_error"] = str(db_error)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis: {str(e)}")


@app.post("/api/register-device")
async def register_device(
    device_id: str = Form(...),
    device_name: str = Form(default="Raspberry Pi"),
    location: str = Form(default="Consultorio")
):
    """Registrar un nuevo dispositivo Raspberry Pi"""
    return {
        "success": True,
        "device_id": device_id,
        "device_name": device_name,
        "location": location,
        "registered_at": datetime.now(pytz.timezone("America/Bogota")).isoformat(),
        "message": "Dispositivo registrado correctamente"
    }


@app.get("/api/devices/{device_id}/stats")
async def get_device_stats(device_id: str):
    """Obtener estadísticas de un dispositivo"""
    # Contar imágenes del dispositivo
    device_images = [f for f in os.listdir(UPLOAD_DIR) if f.startswith(device_id)]
    
    return {
        "device_id": device_id,
        "total_analyses": len(device_images),
        "last_activity": datetime.now(pytz.timezone("America/Bogota")).isoformat()
    }


def get_recommendation(risk_level: str) -> str:
    """Generar recomendación basada en nivel de riesgo"""
    recommendations = {
        "ALTO": "⚠️ URGENTE: Consultar dermatólogo inmediatamente. Se recomienda biopsia.",
        "BAJO": "✅ Lesión aparentemente benigna. Monitorear cambios. Control anual.",
        "REVISAR": "⚡ Resultado inconcluso. Se requiere evaluación dermatológica adicional."
    }
    return recommendations.get(risk_level, "Consultar especialista")


# Para ejecutar con uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8502)
