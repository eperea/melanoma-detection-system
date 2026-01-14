import streamlit as st
import numpy as np
from PIL import Image
import io
from datetime import datetime

# Importar lógica
from logic.sr_model import SRPredictor
from logic.classifier import MelanomaClassifier
from database import init_database, registrar_paciente, guardar_analisis, obtener_historial_paciente, buscar_paciente
from utils.pdf_report import generate_report_pdf

# Configuración de página
st.set_page_config(
    page_title="Sistema de Detección de Melanoma", 
    page_icon="🔬",
    layout="wide"
)

# CSS personalizado para mejor visualización
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .alert-melanoma {
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .alert-nevus {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .patient-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Rutas de modelos
MODEL_SR_PATH = "models/best_srcnn.pth"
MODEL_CL_PATH = "models/keras_model.h5"
LABELS_PATH = "models/labels.txt"

# Inicializar base de datos
try:
    init_database()
except Exception as e:
    st.warning(f"Base de datos no disponible: {e}. El historial no se guardará.")

# Cargar modelos (con caché)
@st.cache_resource
def load_sr_model():
    return SRPredictor(MODEL_SR_PATH)

@st.cache_resource
def load_classifier_model():
    return MelanomaClassifier(MODEL_CL_PATH, LABELS_PATH)

try:
    sr_predictor = load_sr_model()
    classifier = load_classifier_model()
    models_loaded = True
except Exception as e:
    st.error(f"Error cargando modelos: {e}")
    models_loaded = False

# Header principal
st.markdown('<p class="main-header">🔬 Sistema de Detección de Melanoma</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Análisis dermatoscópico asistido por Inteligencia Artificial con Super-Resolución</p>', unsafe_allow_html=True)

# Sidebar - Menú de navegación
st.sidebar.title("📋 Menú")
menu_option = st.sidebar.radio(
    "Seleccione una opción:",
    ["🆕 Nuevo Análisis", "📜 Historial de Paciente", "ℹ️ Acerca de"]
)

# =====================================================
# OPCIÓN 1: NUEVO ANÁLISIS
# =====================================================
if menu_option == "🆕 Nuevo Análisis" and models_loaded:
    
    st.header("📝 Datos del Paciente")
    
    col_form1, col_form2 = st.columns(2)
    
    with col_form1:
        paciente_id = st.text_input("Identificación (Cédula/ID)*", placeholder="Ej: 12345678")
        paciente_nombre = st.text_input("Nombre Completo*", placeholder="Ej: Juan Pérez García")
        paciente_edad = st.number_input("Edad", min_value=0, max_value=120, value=30)
    
    with col_form2:
        paciente_sexo = st.selectbox("Sexo", ["Masculino", "Femenino", "Otro"])
        ubicacion_lesion = st.selectbox("Ubicación de la Lesión", [
            "Espalda", "Brazo derecho", "Brazo izquierdo", 
            "Pierna derecha", "Pierna izquierda", "Tórax", 
            "Abdomen", "Rostro", "Cuello", "Otra"
        ])
        notas_clinicas = st.text_area("Notas Clínicas", placeholder="Observaciones adicionales...")
    
    st.divider()
    st.header("🖼️ Imagen Dermatoscópica")
    
    uploaded_file = st.file_uploader(
        "Subir imagen de la lesión", 
        type=["jpg", "png", "jpeg"],
        help="Formatos aceptados: JPG, PNG, JPEG. Se recomienda imagen de alta calidad."
    )
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        image_original = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Imagen Original")
            st.image(image_original, use_column_width=True)
            st.caption(f"Resolución: {image_original.size[0]}×{image_original.size[1]} px")
        
        # Validar datos del paciente
        datos_completos = paciente_id and paciente_nombre
        
        if not datos_completos:
            st.warning("⚠️ Complete los datos del paciente (Identificación y Nombre) antes de analizar.")
        
        if st.button("🔬 Analizar Imagen", disabled=not datos_completos, type="primary"):
            
            # Barra de progreso
            progress_bar = st.progress(0, text="Iniciando análisis...")
            
            # Paso 1: Super-Resolución
            progress_bar.progress(20, text="Aplicando Super-Resolución...")
            sr_image_array = sr_predictor.predict(file_bytes)
            sr_image_pil = Image.fromarray(sr_image_array)
            
            progress_bar.progress(60, text="Clasificando lesión...")
            
            # Paso 2: Clasificación (usar imagen ORIGINAL, no SR)
            # El modelo fue entrenado con imágenes originales del dataset HAM10000
            original_array = np.array(image_original)
            class_name, confidence, probabilities = classifier.predict(original_array)
            
            progress_bar.progress(90, text="Generando resultados...")
            
            # Mostrar imagen SR
            with col2:
                st.subheader("Imagen Mejorada (SR)")
                st.image(sr_image_pil, use_column_width=True)
                st.caption(f"Resolución: {sr_image_pil.size[0]}×{sr_image_pil.size[1]} px")
            
            progress_bar.progress(100, text="¡Análisis completado!")
            
            # Determinar clase y probabilidades
            is_melanoma = "melanoma" in class_name.lower()
            prob_melanoma = probabilities[0] if len(probabilities) > 0 else 0
            prob_nevus = probabilities[1] if len(probabilities) > 1 else 0
            
            st.divider()
            st.header("📊 Resultados del Diagnóstico")
            
            # Métricas principales
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                st.metric("Diagnóstico", "MELANOMA" if is_melanoma else "NEVUS (Benigno)")
            with col_res2:
                st.metric("Confianza", f"{confidence:.1%}")
            with col_res3:
                st.metric("Fecha", datetime.now().strftime("%d/%m/%Y %H:%M"))
            
            # Probabilidades detalladas
            st.write("**Probabilidades por clase:**")
            col_prob1, col_prob2 = st.columns(2)
            with col_prob1:
                st.progress(float(prob_melanoma), text=f"Melanoma: {prob_melanoma:.2%}")
            with col_prob2:
                st.progress(float(prob_nevus), text=f"Nevus: {prob_nevus:.2%}")
            
            # Alerta según resultado
            if is_melanoma:
                st.markdown("""
                <div class="alert-melanoma">
                    <h3>⚠️ ALERTA - Posible Melanoma Detectado</h3>
                    <p><strong>Recomendación:</strong> Se han detectado patrones compatibles con melanoma. 
                    Se recomienda <strong>derivación inmediata a dermatología oncológica</strong> para biopsia 
                    y evaluación especializada.</p>
                    <p><em>Este resultado es una herramienta de apoyo diagnóstico y no reemplaza el criterio médico profesional.</em></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert-nevus">
                    <h3>✅ Lesión Benigna - Nevus</h3>
                    <p><strong>Recomendación:</strong> La lesión presenta características de un nevus benigno. 
                    Se sugiere <strong>monitoreo periódico</strong> y revisión si hay cambios en tamaño, forma o color.</p>
                    <p><em>Este resultado es una herramienta de apoyo diagnóstico y no reemplaza el criterio médico profesional.</em></p>
                </div>
                """, unsafe_allow_html=True)
            
            # Guardar en base de datos
            try:
                paciente = registrar_paciente(paciente_id, paciente_nombre, paciente_edad, paciente_sexo)
                analisis = guardar_analisis(
                    paciente['id'], 
                    ubicacion_lesion, 
                    notas_clinicas, 
                    class_name, 
                    float(confidence), 
                    float(prob_melanoma), 
                    float(prob_nevus)
                )
                st.success(f"✅ Análisis guardado exitosamente. ID de registro: {analisis['id']}")
            except Exception as e:
                st.warning(f"No se pudo guardar en la base de datos: {e}")
            
            # Generar PDF para descarga
            st.divider()
            st.subheader("📄 Descargar Reporte")
            
            try:
                pdf_bytes = generate_report_pdf(
                    paciente_nombre=paciente_nombre,
                    paciente_id=paciente_id,
                    paciente_edad=paciente_edad,
                    paciente_sexo=paciente_sexo,
                    ubicacion_lesion=ubicacion_lesion,
                    notas_clinicas=notas_clinicas,
                    diagnostico=class_name,
                    confianza=confidence,
                    prob_melanoma=prob_melanoma,
                    prob_nevus=prob_nevus
                )
                
                # Nombre del archivo con fecha
                fecha_archivo = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"Reporte_Melanoma_{paciente_id}_{fecha_archivo}.pdf"
                
                st.download_button(
                    label="📥 Descargar Reporte PDF",
                    data=pdf_bytes,
                    file_name=nombre_archivo,
                    mime="application/pdf",
                    type="primary"
                )
                st.caption("El reporte incluye todos los datos del paciente y resultados del análisis.")
            except Exception as e:
                st.error(f"Error generando PDF: {e}")
            
            # Botón para limpiar y nuevo análisis
            if st.button("🔄 Realizar Nuevo Análisis"):
                st.rerun()

# =====================================================
# OPCIÓN 2: HISTORIAL DE PACIENTE
# =====================================================
elif menu_option == "📜 Historial de Paciente":
    st.header("📜 Historial de Análisis")
    
    buscar_id = st.text_input("Ingrese la Identificación del Paciente:", placeholder="Ej: 12345678")
    
    if st.button("🔍 Buscar") and buscar_id:
        try:
            paciente = buscar_paciente(buscar_id)
            
            if paciente:
                st.markdown(f"""
                <div class="patient-card">
                    <h3>👤 {paciente['nombre']}</h3>
                    <p><strong>ID:</strong> {paciente['identificacion']} | 
                    <strong>Edad:</strong> {paciente['edad']} años | 
                    <strong>Sexo:</strong> {paciente['sexo']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                historial = obtener_historial_paciente(buscar_id)
                
                if historial:
                    st.write(f"**Total de análisis:** {len(historial)}")
                    
                    for i, analisis in enumerate(historial, 1):
                        is_melanoma = "melanoma" in analisis['diagnostico'].lower()
                        color = "🔴" if is_melanoma else "🟢"
                        
                        with st.expander(f"{color} Análisis #{i} - {analisis['fecha_analisis'].strftime('%d/%m/%Y %H:%M')}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Diagnóstico:** {analisis['diagnostico']}")
                                st.write(f"**Confianza:** {analisis['confianza']:.1%}")
                                st.write(f"**Ubicación:** {analisis['ubicacion_lesion']}")
                            with col2:
                                st.write(f"**P(Melanoma):** {analisis['probabilidad_melanoma']:.2%}")
                                st.write(f"**P(Nevus):** {analisis['probabilidad_nevus']:.2%}")
                            if analisis['notas_clinicas']:
                                st.write(f"**Notas:** {analisis['notas_clinicas']}")
                else:
                    st.info("No hay análisis registrados para este paciente.")
            else:
                st.warning("Paciente no encontrado. Verifique la identificación.")
        except Exception as e:
            st.error(f"Error consultando historial: {e}")

# =====================================================
# OPCIÓN 3: ACERCA DE
# =====================================================
elif menu_option == "ℹ️ Acerca de":
    st.header("ℹ️ Acerca del Sistema")
    
    st.markdown("""
    ### 🔬 Sistema de Detección de Melanoma con Super-Resolución
    
    Este sistema utiliza técnicas avanzadas de Inteligencia Artificial para asistir en el 
    diagnóstico temprano de melanoma a partir de imágenes dermatoscópicas.
    
    #### 🧠 Tecnología Utilizada
    
    1. **Super-Resolución Convolucional (SRCNN)**
       - Mejora la calidad y resolución de las imágenes
       - Permite identificar detalles que podrían pasar desapercibidos
       - Implementado con PyTorch
    
    2. **Clasificación por Deep Learning**
       - Red neuronal entrenada para distinguir entre Melanoma y Nevus
       - Implementado con TensorFlow/Keras
    
    #### ⚠️ Aviso Importante
    
    Este sistema es una **herramienta de apoyo diagnóstico** y no sustituye la evaluación 
    de un profesional médico especializado. Los resultados deben ser siempre validados 
    por un dermatólogo.
    
    ---
    
    **Proyecto de Tesis**
    
    *Desarrollado como parte de la investigación en detección temprana de cáncer de piel 
    mediante técnicas de visión por computadora e inteligencia artificial.*
    """)

# Footer
st.sidebar.divider()
st.sidebar.caption("© 2026 - Sistema de Detección de Melanoma")
st.sidebar.caption("Proyecto de Tesis")
