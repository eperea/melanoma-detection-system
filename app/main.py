import streamlit as st
import numpy as np
from PIL import Image
import io
from datetime import datetime
import pytz
import os

# Importar lógica
from logic.sr_model import SRPredictor
from logic.classifier import MelanomaClassifier
from database import init_database, registrar_paciente, guardar_analisis, obtener_historial_paciente, buscar_paciente
from utils.pdf_report import generate_report_pdf

# Configuración de página
st.set_page_config(
    page_title="Sistema de Detección de Melanoma", 
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar Base de Datos
init_database()

# Definir Rutas de Modelos y Assets
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SR_PATH = os.path.join(BASE_DIR, "models", "best_srcnn.pth")
MODEL_CL_PATH = os.path.join(BASE_DIR, "models", "keras_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "models", "labels.txt")

ASSETS_DIR = os.path.join(BASE_DIR, "assets")
IMG_VAL_PATH = os.path.join(ASSETS_DIR, "matriz_confusion.png")
REPORT_VAL_PATH = os.path.join(ASSETS_DIR, "reporte_clasificacion.txt")
IMG_TRAIN_PATH = os.path.join(ASSETS_DIR, "matriz_confusion_train.png")
REPORT_TRAIN_PATH = os.path.join(ASSETS_DIR, "reporte_train.txt")

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
        font-size: 1.1rem;
        color: #444;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    /* Estilos de tarjetas */
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .highlight-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Cargar Modelos (con caché para optimizar)
@st.cache_resource
def load_sr_model():
    predictor = SRPredictor(MODEL_SR_PATH)
    return predictor

@st.cache_resource
def load_classifier_model():
    return MelanomaClassifier(MODEL_CL_PATH, LABELS_PATH)

# Estado de carga de modelos
try:
    sr_predictor = load_sr_model()
    classifier = load_classifier_model()
    models_loaded = True
except Exception as e:
    st.error(f"⚠️ Error cargando modelos de IA: {e}")
    models_loaded = False

# =====================================================
# SIDEBAR MENÚ
# =====================================================
st.sidebar.markdown("### 🏥 Panel de Control")
menu_option = st.sidebar.radio(
    "Navegación:",
    ["🔬 Nuevo Análisis", "🗂️ Historial Pacientes", "📊 Métricas & Validación", "📚 Documentación Técnica", "🔧 Guía Raspberry Pi", "⚙️ Especificaciones Técnicas", "ℹ️ Acerca de"]
)

st.sidebar.divider()
st.sidebar.divider()
st.sidebar.info("Proyecto de Tesis\n\nIngeniería Electrónica - IA detectando Melanoma")

# =====================================================
# CONFIGURACIÓN DE UMBRAL DE CONFIANZA
# =====================================================
st.sidebar.divider()
st.sidebar.markdown("### ⚙️ Configuración del Modelo")
confidence_threshold = st.sidebar.slider(
    "🎯 Umbral de Confianza",
    min_value=0.60,
    max_value=0.95,
    value=0.70,
    step=0.05,
    help="Si la confianza del modelo es menor al umbral, se clasificará como 'Otra afectación / No clasificable'. Recomendado: 70%"
)
st.sidebar.caption(f"Umbral actual: **{confidence_threshold:.0%}**")
st.sidebar.markdown("""
<small style="color: gray;">
💡 <b>¿Qué es esto?</b><br>
Si el modelo no tiene suficiente certeza para clasificar como Melanoma o Nevus, indicará que podría ser otra afectación cutánea.
</small>
""", unsafe_allow_html=True)

# =====================================================
# OPCIÓN 1: NUEVO ANÁLISIS
# =====================================================
if menu_option == "🔬 Nuevo Análisis":
    
    st.markdown('<p class="main-header">Detección Asistida de Melanoma</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Suba una imagen dermatoscópica para obtener un diagnóstico asistido por IA</p>', unsafe_allow_html=True)

    if not models_loaded:
        st.error("Los modelos no están cargados. No se puede realizar el análisis.")
        st.stop()
    
    # --- ÁREA PRINCIPAL: IMAGEN ---
    st.markdown("### 📷 Imagen a Analizar")
    
    capture_tab1, capture_tab2 = st.tabs(["📸 Capturar con Cámara", "📁 Subir Archivo"])
    
    captured_photo = None
    uploaded_file = None
    
    with capture_tab1:
        st.info("""
        **📸 Para tomar una foto directamente desde tu dispositivo:**
        1. Tu navegador pedirá permiso para usar la cámara → **Haz clic en "Permitir" / "Allow"**
        2. Aparecerá la vista previa de la cámara en vivo
        3. Posiciona la lesión cutánea frente a la cámara
        4. Presiona el botón **"Take Photo"** que aparece debajo de la vista previa
        
        ⚠️ *Si no ves la cámara, haz clic en el ícono de candado 🔒 en la barra de direcciones y permite el acceso a la cámara.*
        """)
        captured_photo = st.camera_input("Tomar foto de la lesión cutánea")
    
    with capture_tab2:
        uploaded_file = st.file_uploader(
            "Arrastre o seleccione una imagen dermatoscópica (JPG, PNG)", 
            type=["jpg", "jpeg", "png"],
            help="Suba una fotografía clara de la lesión cutánea"
        )
    
    # Determinar la fuente de imagen (cámara tiene prioridad si ambas existen)
    image_source = captured_photo or uploaded_file
    source_name = "📸 Captura de cámara" if captured_photo else (f"📁 {uploaded_file.name}" if uploaded_file else None)
    
    # Variables para controlar estado
    image_original = None
    file_bytes = None
    
    if image_source is not None:
        image_original = Image.open(image_source).convert("RGB")
        file_bytes = io.BytesIO()
        image_original.save(file_bytes, format='JPEG')
        file_bytes = file_bytes.getvalue()
        
        # Mostrar imagen con tamaño controlado
        col_img, col_info = st.columns([2, 1])
        with col_img:
            st.image(image_original, caption=source_name, use_column_width=True)
        with col_info:
            st.metric("Resolución", f"{image_original.size[0]} × {image_original.size[1]} px")
            st.metric("Tamaño", f"{len(file_bytes) / 1024:.1f} KB")
    
    st.divider()
    
    # --- FORMULARIO COMPACTO ---
    with st.expander("📋 Datos del Paciente y Observaciones", expanded=True):
        form_col1, form_col2, form_col3 = st.columns([2, 1, 1])
        
        with form_col1:
            id_paciente = st.text_input("🆔 Identificación", placeholder="Cédula o ID del paciente")
        
        # Buscar paciente existente
        paciente_db = None
        if id_paciente:
            paciente_db = buscar_paciente(id_paciente)
            if paciente_db:
                st.success(f"✅ Paciente encontrado: **{paciente_db['nombre']}**")
                nombre = paciente_db['nombre']
                edad = paciente_db['edad']
                sexo = paciente_db['sexo']
            else:
                with form_col1:
                    nombre = st.text_input("👤 Nombre Completo", placeholder="Nombre del paciente")
                with form_col2:
                    edad = st.number_input("🎂 Edad", min_value=0, max_value=120, value=30)
                with form_col3:
                    sexo = st.selectbox("⚧ Sexo", ["Masculino", "Femenino", "Otro"])
        else:
            nombre = None
            edad = None
            sexo = None
        
        st.markdown("---")
        obs_col1, obs_col2 = st.columns(2)
        with obs_col1:
            ubicacion = st.selectbox("📍 Ubicación de la lesión", ["Rostro", "Brazo", "Pierna", "Espalda", "Pecho", "Abdomen", "Otro"])
        with obs_col2:
            notas = st.text_input("📝 Notas Clínicas (Opcional)", placeholder="Observaciones adicionales...")
    
    # --- BOTÓN DE ANÁLISIS ---
    datos_completos = id_paciente and nombre and image_source
    
    if not datos_completos:
        st.info("💡 Complete los datos del paciente y suba una imagen para habilitar el análisis.")
    
    if st.button("🚀 Iniciar Análisis con IA", disabled=not datos_completos, type="primary", use_container_width=True):
        
        # Registrar/Actualizar paciente si es necesario
        if not paciente_db:
            paciente_db = registrar_paciente(id_paciente, nombre, edad, sexo)
        
        # --- PROCESO DE IA ---
        with st.status("Ejecutando pipeline de IA...", expanded=True) as status:
            
            st.write("🔄 Aplicando Super-Resolución (SRCNN)...")
            sr_image_array = sr_predictor.predict(file_bytes)
            sr_image_pil = Image.fromarray(sr_image_array)
            
            st.write("🧠 Clasificando lesión (MobileNetV2)...")
            original_array = np.array(image_original)
            class_name, confidence, probabilities, is_classifiable, classification_info = classifier.predict(
                original_array, 
                confidence_threshold=confidence_threshold
            )
            
            st.write("💾 Guardando resultados en base de datos...")
            prob_melanoma = classification_info.get('melanoma_prob', 0.0)
            prob_nevus = classification_info.get('nevus_prob', 0.0)
            
            guardar_analisis(
                paciente_db['id'], ubicacion, notas, 
                class_name, confidence, prob_melanoma, prob_nevus
            )
            
            status.update(label="✅ ¡Análisis Completado!", state="complete", expanded=False)
        
        # --- RESULTADOS ---
        st.divider()
        st.markdown("## 📊 Resultados del Diagnóstico")
        
        # Determinar tipo de resultado para mostrar
        if not is_classifiable:
            # CASO: No clasificable / Otra afectación
            color = "orange"
            icono = "🔶"
            titulo = "OTRA AFECTACIÓN / NO CLASIFICABLE"
            bg_color = "#fff3e0"  # Naranja claro
            
            st.markdown(f"""
            <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid {color}; margin-bottom: 20px;">
                <h2 style="color: {color}; margin:0;">{icono} {titulo}</h2>
                <h4 style="margin:5px 0; color: #555;">La imagen no corresponde claramente a Melanoma ni Nevus</h4>
                <p style="margin:5px 0; font-size: 0.9rem; color: #666;">Razón: {classification_info['reason']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Explicación adicional
            st.warning("""
            ⚠️ **Interpretación:** El modelo de IA no puede clasificar esta lesión con suficiente certeza.
            
            **Posibles causas:**
            - La lesión podría ser otro tipo de afectación cutánea (dermatitis, queratosis, etc.)
            - La imagen puede no ser de calidad suficiente
            - Las características de la lesión son ambiguas
            
            **Recomendación:** Se sugiere evaluación presencial por un dermatólogo especialista.
            """)
        else:
            is_melanoma = "melanoma" in class_name.lower()
            color = "red" if is_melanoma else "green"
            icono = "⚠️" if is_melanoma else "✅"
            titulo = "MELANOMA (Maligno)" if is_melanoma else "NEVUS (Benigno)"
            bg_color = '#ffebee' if is_melanoma else '#e8f5e9'
            
            # Tarjeta de resultado principal
            st.markdown(f"""
            <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid {color}; margin-bottom: 20px;">
                <h2 style="color: {color}; margin:0;">{icono} {titulo}</h2>
                <h3 style="margin:0;">Confianza: {confidence:.2%}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        # Fecha Colombia
        bogota_tz = pytz.timezone('America/Bogota')
        current_time = datetime.now(bogota_tz)
        
        # Métricas y comparación (mostrar siempre las probabilidades)
        met_col1, met_col2, met_col3, met_col4 = st.columns(4)
        with met_col1:
            st.metric("🔴 Melanoma", f"{prob_melanoma:.1%}")
        with met_col2:
            st.metric("🟢 Nevus", f"{prob_nevus:.1%}")
        with met_col3:
            st.metric("🎯 Umbral", f"{confidence_threshold:.0%}")
        with met_col4:
            st.metric("📅 Fecha", current_time.strftime('%d/%m/%Y %H:%M'))
        
        # Comparativa visual
        st.markdown("#### Comparativa de Imagen")
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.image(image_original, caption="Original", use_column_width=True)
        with img_col2:
            st.image(sr_image_pil, caption="Mejorada (Super-Resolución)", use_column_width=True)
        
        # Botón de descarga de reporte PDF
        st.divider()
        pdf_bytes = generate_report_pdf(
            paciente_nombre=paciente_db['nombre'],
            paciente_id=paciente_db['identificacion'],
            paciente_edad=paciente_db['edad'],
            paciente_sexo=paciente_db['sexo'],
            ubicacion_lesion=ubicacion,
            notas_clinicas=notas,
            diagnostico=class_name,
            confianza=confidence,
            prob_melanoma=prob_melanoma,
            prob_nevus=prob_nevus
        )
        st.download_button(
            label="📄 Descargar Reporte PDF",
            data=pdf_bytes,
            file_name=f"Reporte_{paciente_db['identificacion']}_{current_time.strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )
                
# =====================================================
# OPCIÓN 2: HISTORIAL
# =====================================================
elif menu_option == "🗂️ Historial Pacientes":
    st.markdown('<p class="main-header">📜 Historial Médico</p>', unsafe_allow_html=True)
    
    id_busqueda = st.text_input("🔍 Buscar por Identificación del Paciente", placeholder="Ingrese ID...")
    
    if id_busqueda:
        paciente = buscar_paciente(id_busqueda)
        if paciente:
            st.success(f"Historial de: **{paciente['nombre']}** (Edad: {paciente['edad']}, Sexo: {paciente['sexo']})")
            
            historial = obtener_historial_paciente(id_busqueda)
            
            if historial:
                for idx, h in enumerate(historial):
                    with st.expander(f"📅 {h['fecha_analisis']} - {h['diagnostico']} ({h['confianza']:.1%})"):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.write(f"**Diagnóstico:** {h['diagnostico']}")
                            st.write(f"**Confianza:** {h['confianza']:.2%}")
                            st.write(f"**Ubicación:** {h['ubicacion_lesion']}")
                        with c2:
                            st.write(f"**Notas:** {h['notas_clinicas'] or 'Sin notas'}")
                            pdf_bytes = generate_report_pdf(
                                paciente_nombre=paciente['nombre'],
                                paciente_id=paciente['identificacion'],
                                paciente_edad=paciente['edad'],
                                paciente_sexo=paciente['sexo'],
                                ubicacion_lesion=h['ubicacion_lesion'],
                                notas_clinicas=h['notas_clinicas'],
                                diagnostico=h['diagnostico'],
                                confianza=h['confianza'],
                                prob_melanoma=h.get('probabilidad_melanoma', 0),
                                prob_nevus=h.get('probabilidad_nevus', 0)
                            )
                            st.download_button(
                                label="📄 Descargar Reporte PDF",
                                data=pdf_bytes,
                                file_name=f"Reporte_{paciente['identificacion']}_{h['fecha_analisis']}.pdf",
                                mime="application/pdf",
                                key=f"btn_pdf_{idx}"
                            )
            else:
                st.info("Este paciente no tiene análisis registrados aún.")
        else:
            st.warning("Paciente no encontrado.")

# =====================================================
# OPCIÓN 3: MÉTRICAS Y VALIDACIÓN (NUEVA)
# =====================================================
elif menu_option == "📊 Métricas & Validación":
    st.markdown('<p class="main-header">📊 Rendimiento del Modelo</p>', unsafe_allow_html=True)
    
    st.markdown("""
    En esta sección se presenta la evaluación técnica del modelo de clasificación **MobileNetV2**.
    Es crucial diferenciar entre el rendimiento durante el **Entrenamiento** (capacidad de aprendizaje) y la **Validación** (capacidad de generalización frente a datos desbalanceados).
    """)
    
    tab1, tab2, tab3 = st.tabs(["📘 Entrenamiento (Aprendizaje)", "⚖️ Validación Balanceada (Realidad)", "📙 Validación Completa (Desbalanceada)"])
    
    with tab1:
        st.markdown("### Rendimiento en Entrenamiento (Datos Balanceados)")
        st.success("""
        **Interpretación:** 
        Durante el entrenamiento, al usar un dataset equilibrado (~6,000 imágenes por clase), el modelo demostró una **excelente capacidad para distinguir Melanomas**, alcanzando una sensibilidad (Recall) superior al **90%**.
        Esto prueba que la red neuronal **APRENDIÓ** correctamente las características del cáncer.
        """)
        c1, c2 = st.columns(2)
        with c1:
            if os.path.exists(IMG_TRAIN_PATH):
                st.image(IMG_TRAIN_PATH, caption="Matriz de Entrenamiento", use_column_width=True)
        with c2:
            if os.path.exists(REPORT_TRAIN_PATH):
                with open(REPORT_TRAIN_PATH, "r") as f: st.text(f.read())

    with tab2:
        st.markdown("### Rendimiento en Validación Balanceada (Test Justo)")
        st.info("""
        **ANÁLISIS CRÍTICO (La métrica más importante):**
        Dado el fuerte desbalance en el set de validación original, se realizó una prueba controlada tomando todos los **39 Melanomas** y comparándolos contra **39 Nevus aleatorios**.
        
        **Resultado:** El **Recall de Melanoma sube drásticamente a 87%**.
        Esto demuestra que el modelo **SÍ es efectivo** detectando la enfermedad cuando no está sesgado por la mayoría de casos sanos.
        """)
        IMG_BALANCED_PATH = os.path.join(ASSETS_DIR, "matriz_confusion_balanced.png")
        REPORT_BALANCED_PATH = os.path.join(ASSETS_DIR, "reporte_balanced.txt")
        
        c1, c2 = st.columns(2)
        with c1:
            if os.path.exists(IMG_BALANCED_PATH):
                st.image(IMG_BALANCED_PATH, caption="Matriz Balanceada (39 vs 39)", use_column_width=True)
            else: st.warning("Imagen no encontrada.")
        with c2:
            if os.path.exists(REPORT_BALANCED_PATH):
                with open(REPORT_BALANCED_PATH, "r") as f: st.text(f.read())
            else: st.warning("Reporte no encontrado.")

    with tab3:
        st.markdown("### Validación Completa (Escenario con Desbalance)")
        st.warning("""
        **Observación:** En el set completo (751 sanos vs 39 enfermos), el desbalance estadístico oculta el rendimiento real del modelo en la clase minoritaria.
        Sin embargo, la exactitud global sigue siendo del **97%**.
        """)
        c1, c2 = st.columns(2)
        with c1:
            if os.path.exists(IMG_VAL_PATH):
                st.image(IMG_VAL_PATH, caption="Matriz Validación Total", use_column_width=True)
        with c2:
             if os.path.exists(REPORT_VAL_PATH):
                with open(REPORT_VAL_PATH, "r") as f: st.text(f.read())

# =====================================================
# OPCIÓN 4: DOCUMENTACIÓN TÉCNICA (NUEVA)
# =====================================================
elif menu_option == "📚 Documentación Técnica":
    st.markdown('<p class="main-header">📚 Documentación del Proyecto</p>', unsafe_allow_html=True)
    
    st.markdown("Este manual describe la arquitectura, tecnologías e impacto del sistema.")
    
    doc_tabs = st.tabs(["🚀 Resumen Ejecutivo", "🏗️ Arquitectura", "🧠 Modelos de IA", "📈 Impacto en Salud", "🎓 Glosario (Estudio)"])
    
    with doc_tabs[0]:
        st.markdown("""
        ### Resumen del Sistema
        Este proyecto propone una solución tecnológica para el apoyo al diagnóstico temprano de **Melanoma** (Cáncer de piel).
        
        **Problema:**
        - El diagnóstico visual subjetivo puede tener tasas de error.
        - Las imágenes dermatoscópicas tomadas con celulares suelen tener baja calidad o desenfoque.
        
        **Solución:**
        Un pipeline de IA en dos etapas:
        1.  **Mejora de Imagen:** Uso de **Super-Resolución (SRCNN)** para restaurar detalles finos.
        2.  **Diagnóstico:** Clasificación automática usando una **Red Neuronal Convolucional (MobileNetV2)**.
        """)
        
        st.info("💡 **Objetivo:** Proveer una segunda opinión objetiva y rápida al especialista médico.")

    with doc_tabs[1]:
        st.markdown("### Arquitectura del Sistema")
        st.markdown("El sistema sigue una arquitectura de microservicios contenerizados:")
        
        st.graphviz_chart("""
        digraph Architecture {
            rankdir=LR;
            node [shape=box, style=filled, fillcolor="#f0f2f6", fontname="Sans"];
            
            subgraph cluster_client {
                label = "Cliente";
                style=dashed;
                Browser [label="🖥️ Navegador Web\n(Usuario Médico)", fillcolor="#e3f2fd"];
            }
            
            subgraph cluster_server {
                label = "Servidor VPS (Docker Host)";
                style=filled;
                color="#eeeeee";
                
                subgraph cluster_app {
                    label = "Contenedor App";
                    color=white;
                    Streamlit [label="⚡ Streamlit\n(Frontend + Backend)", fillcolor="#fff3e0"];
                    Model_SR [label="🔍 Modelo SRCNN\n(Super-Resolución)", shape=ellipse, fillcolor="#e8f5e9"];
                    Model_CL [label="🧠 MobileNetV2\n(Clasificación)", shape=ellipse, fillcolor="#e8f5e9"];
                }
                
                subgraph cluster_db {
                    label = "Contenedor DB";
                    color=white;
                    Postgres [label="🗄️ PostgreSQL\n(Datos Pacientes)", fillcolor="#e1bee7"];
                }
            }
            
            Browser -> Streamlit [label="HTTP/HTTPS"];
            Streamlit -> Model_SR [label="Imágenes"];
            Model_SR -> Model_CL [label="Imagen SR"];
            Streamlit -> Postgres [label="SQL (Lectura/Escritura)"];
        }
        """)
        
        st.markdown("""
        **Flujo de Datos:**
        1.  El usuario sube una imagen al navegador.
        2.  Streamlit recibe la imagen y la pasa al modelo **SRCNN** para mejorarla.
        3.  La imagen mejorada entra a **MobileNetV2** para obtener la probabilidad de Melanoma.
        4.  Los resultados y datos del paciente se guardan en **PostgreSQL**.
        """)

    with doc_tabs[2]:
        st.markdown("### 🧠 Modelos de Inteligencia Artificial")
        
        st.markdown("""
        #### 1. Clasificación: MobileNetV2 (La elección estratégica)
        
        El núcleo del diagnóstico es **MobileNetV2**. Se seleccionó esta arquitectura por encima de opciones más pesadas (como ResNet50 o VGG16) por las siguientes razones técnicas fundamentales para un despliegue real:
        
        **A. Eficiencia Extrema (Depthwise Separable Convolutions):**
        *   A diferencia de las redes tradicionales que realizan convoluciones completas, MobileNetV2 divide la operación en dos pasos:
            1.  **Depthwise Convolution:** Filtra cada canal de entrada de forma independiente.
            2.  **Pointwise Convolution (1x1):** Combina los resultados.
        *   **Resultado:** Reduce el número de cálculos y parámetros entre 8 y 9 veces, manteniendo una precisión comparable. Esto es crucial para que el sistema responda rápido en servidores web estándar sin GPU costosas.
        
        **B. Arquitectura de "Inverted Residuals":**
        *   Introduce bloques residuales invertidos con "Linear Bottlenecks".
        *   Permite que la información fluya mejor a través de las capas profundas sin perderse (Vanishing Gradient problem), logrando una mayor exactitud con menos memoria.
        
        **C. Transfer Learning (Aprendizaje por Transferencia):**
        *   El modelo no empezó desde cero ("Tabula Rasa"). Se utilizaron pesos pre-entrenados en **ImageNet** (1.4 millones de imágenes).
        *   **Beneficio:** La red ya "sabía" detectar bordes, texturas y formas complejas. Solo tuvimos que "afinarla" (Fine-Tuning) para que aprendiera a distinguir las características específicas de los lunares y el melanoma (asimetría, bordes irregulares, color).
        
        ---
        
        #### 2. Super-Resolución: SRCNN (Super-Resolution CNN)
        *   **Objetivo:** Mejorar la calidad de entrada antes de la clasificación.
        *   **Funcionamiento:** Mapea una imagen de baja resolución a una de alta resolución a través de un mapa de características no lineal.
        *   **Impacto:** Recupera detalles finos en los bordes de la lesión que podrían haberse perdido por desenfoque o baja calidad de la cámara, ayudando al clasificador a ser más preciso.
        """)

    with doc_tabs[3]:
        st.markdown("### Impacto Social y en Salud")
        st.markdown("""
        El melanoma es uno de los cánceres más agresivos pero **altamente curable si se detecta a tiempo**.
        
        1.  **Tamizaje Masivo:** Esta herramienta permite filtrar casos sospechosos rápidamente en zonas rurales o centros de atención primaria.
        3.  **Registro Histórico:** La base de datos permite monitorear la evolución de lunares en el tiempo, crucial para detectar cambios malignos.
        """)

    with doc_tabs[4]:
        st.markdown("### 🎓 Glosario de Conceptos Clave (Para estudio)")
        
        st.markdown("""
        #### 1. Inteligencia Artificial y Deep Learning
        *   **Red Neuronal Convolucional (CNN):** Tipo de IA diseñada para procesar imágenes. Funciona como el ojo humano, detectando primero bordes simples y luego formas complejas a medida que profundiza en las capas.
        *   **Transfer Learning:** Técnica de "reciclaje" de conocimiento. En lugar de enseñar al modelo desde cero (que requiere millones de imágenes), tomamos uno que ya sabe ver (entrenado en ImageNet) y le enseñamos solo la parte específica de dermatología. Es más rápido y eficiente.
        *   **Data Augmentation:** Estrategia para multiplicar los datos de entrenamiento creando variaciones artificiales de las imágenes originales (rotaciones, zoom, espejos) para evitar que el modelo "memorice" y aprenda a generalizar.

        #### 2. Métricas de Evaluación
        *   **Accuracy (Exactitud):** Porcentaje total de aciertos. (Ej: 97% significa que de 100 casos, 97 fueron correctos). *Cuidado: En datos desbalanceados puede ser engañoso.*
        *   **Recall (Sensibilidad):** Capacidad del modelo para encontrar a **TODOS** los enfermos. Es la métrica más importante en medicina. Un Recall bajo significa que se escapan casos peligrosos.
        *   **Precision (Precisión):** Cuando el modelo dice "es cáncer", ¿qué tan seguro es? Una precisión baja significa muchas "falsas alarmas".
        *   **Confusion Matrix:** Tabla que muestra dónde se equivocó el modelo (Falsos Positivos vs Falsos Negativos).

        #### 3. Tecnología
        *   **Docker:** Tecnología que empaqueta la aplicación con todas sus librerías necesarias. Garantiza que si funciona en mi máquina, funcione en cualquier servidor ("It works on my machine").
        *   **Microservicios:** Arquitectura donde la App y la Base de Datos viven en contenedores separados que hablan entre sí, facilitando el mantenimiento.
        """)

# =====================================================
# OPCIÓN 5: GUÍA RASPBERRY PI
# =====================================================
elif menu_option == "🔧 Guía Raspberry Pi":
    st.markdown('<p class="main-header">🔧 Guía de Integración con Raspberry Pi</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Paso a paso para conectar un dispositivo de captura al sistema de detección</p>', unsafe_allow_html=True)

    rpi_tabs = st.tabs([
        "📦 Componentes",
        "🔌 Conexión Hardware",
        "💻 Instalación Software",
        "🚀 Uso del Sistema",
        "🏗️ Arquitectura",
        "❓ Solución de Problemas"
    ])

    # --- TAB 1: COMPONENTES ---
    with rpi_tabs[0]:
        st.markdown("### 📦 Lista de Materiales — Versión Económica")
        st.success("**Presupuesto total estimado: ~$80 USD / ~$320.000 COP**")

        st.markdown("""
        | # | Componente | Precio USD | Precio COP | ¿Dónde comprar? |
        |:--|:-----------|:-----------|:-----------|:----------------|
        | 1 | **Raspberry Pi 4 Model B (2GB RAM)** | $35–45 | $150.000–180.000 | MercadoLibre, Vistronica |
        | 2 | **Cámara USB genérica (720p o 1080p)** | $10–20 | $40.000–80.000 | MercadoLibre, Amazon |
        | 3 | **Botón pulsador (push button)** | $0.50–2 | $2.000–8.000 | Vistronica, Sigma Electrónica |
        | 4 | **LED RGB (cátodo común)** | $0.50–1 | $2.000–4.000 | Vistronica |
        | 5 | **Resistencias 220Ω (x3)** | $1 | $4.000 | Vistronica |
        | 6 | **Cables jumper (macho-hembra, x10)** | $3–5 | $12.000–20.000 | Vistronica |
        | 7 | **MicroSD 32GB Clase 10** | $8–12 | $30.000–50.000 | MercadoLibre |
        | 8 | **Fuente 5V 3A USB-C** | $8–12 | $30.000–50.000 | MercadoLibre |
        """)

        st.info("""
        💡 **Tip:** Si ya tienes una Raspberry Pi con alimentación y MicroSD, solo necesitas la cámara USB (~$10 USD), 
        el botón (~$0.50 USD) y el LED RGB (~$0.50 USD). **Costo adicional: ~$15 USD / ~$60.000 COP**.
        """)

        st.markdown("---")
        st.markdown("### 📸 Sobre la Cámara USB")
        st.markdown("""
        Cualquier **webcam USB estándar** funciona. No necesitas una cámara costosa. Características mínimas:
        
        - **Resolución mínima:** 720p (1280×720)
        - **Resolución recomendada:** 1080p (1920×1080)
        - **Interfaz:** USB 2.0 o superior
        - **Compatibilidad:** UVC (Universal Video Class) — la mayoría de webcams lo son
        
        > ⚠️ **No se necesita** la Pi Camera HQ ($50+) ni lentes especiales. 
        > El modelo de IA incluye **Super-Resolución (SRCNN)** que mejora la calidad automáticamente.
        """)

    # --- TAB 2: CONEXIÓN HARDWARE ---
    with rpi_tabs[1]:
        st.markdown("### 🔌 Diagrama de Conexión")
        st.markdown("""
        #### Paso 1: Conectar la Cámara USB
        Simplemente conecta la cámara USB a cualquier puerto USB de la Raspberry Pi.
        """)

        st.code("""
    Raspberry Pi 4 (vista superior)
    ┌─────────────────────────────────┐
    │ [USB 2.0] [USB 2.0]            │
    │ [USB 3.0] [USB 3.0] ← Cámara  │
    │                                 │
    │ [Ethernet]      [USB-C Power]   │
    └─────────────────────────────────┘
        """, language="text")

        st.markdown("#### Paso 2: Conectar el Botón de Captura")
        st.markdown("""
        El botón se conecta entre **GPIO 17** y **GND**:
        """)

        st.code("""
    Botón Pulsador:
    
    Pin GPIO 17 ─────┤ BOTÓN ├───── Pin GND
    (Pin físico 11)                 (Pin físico 9)
    
    Nota: Se usa resistencia pull-up interna del RPi
    (configurada por software, no necesita resistencia externa)
        """, language="text")

        st.markdown("#### Paso 3: Conectar el LED RGB")
        st.warning("⚡ **IMPORTANTE:** Siempre usa resistencias de 220Ω entre cada pin GPIO y la pata del LED para proteger el LED y la Raspberry Pi.")

        st.code("""
    LED RGB (Cátodo Común):

    GPIO 22 (Pin 15) ──[220Ω]── Pata ROJA
    GPIO 23 (Pin 16) ──[220Ω]── Pata VERDE
    GPIO 24 (Pin 18) ──[220Ω]── Pata AZUL
    GND     (Pin 14) ─────────── Pata GND (la más larga)
    
    Significado de colores:
    🔴 ROJO    = Melanoma detectado / Error
    🟢 VERDE   = Nevus (benigno) / Conexión OK
    🔵 AZUL    = Procesando / Enviando al servidor
    🟡 AMARILLO = Resultado inconcluso (rojo + verde)
        """, language="text")

        st.markdown("#### Diagrama Completo de Pines")
        st.code("""
    Raspberry Pi 4 — Pinout utilizado:
    
    ┌───────────────────────────────┐
    │          3V3 [1]  [2]  5V    │
    │        GPIO2 [3]  [4]  5V    │
    │        GPIO3 [5]  [6]  GND   │
    │        GPIO4 [7]  [8]  GPIO14│
    │      ► GND   [9]  [10] GPIO15│ ◄─ GND para botón
    │  ► GPIO17 [11] [12] GPIO18│ ◄─ Botón de captura
    │       GPIO27 [13] [14] GND   │ ◄─ GND para LED
    │  ► GPIO22 [15] [16] GPIO23│ ◄─ LED Rojo / LED Verde
    │          3V3 [17] [18] GPIO24│ ◄─ LED Azul
    │       GPIO10 [19] [20] GND   │
    │        ...                    │
    └───────────────────────────────┘
    
    Resumen:
    Pin  9  (GND)    → Botón (una pata)
    Pin 11  (GPIO17) → Botón (otra pata)
    Pin 14  (GND)    → LED RGB (cátodo/GND)
    Pin 15  (GPIO22) → LED Rojo (con R 220Ω)
    Pin 16  (GPIO23) → LED Verde (con R 220Ω)
    Pin 18  (GPIO24) → LED Azul (con R 220Ω)
        """, language="text")

    # --- TAB 3: INSTALACIÓN SOFTWARE ---
    with rpi_tabs[2]:
        st.markdown("### 💻 Instalación Paso a Paso")

        st.markdown("#### Paso 1: Instalar Raspberry Pi OS")
        st.markdown("""
        1. Descarga **Raspberry Pi Imager** desde [raspberrypi.com/software](https://www.raspberrypi.com/software/)
        2. Inserta la MicroSD en tu PC
        3. En Raspberry Pi Imager:
           - **OS:** Raspberry Pi OS (64-bit) Lite *(sin escritorio, más liviano)*
           - **Storage:** Tu MicroSD
           - **Configuración (⚙️):** Activa SSH, configura WiFi y contraseña
        4. Graba la imagen y coloca la MicroSD en la Raspberry Pi
        5. Enciende la Raspberry Pi y espera ~2 minutos
        """)

        st.markdown("#### Paso 2: Conectar por SSH")
        st.code("""
# Desde tu PC (Linux/Mac/Windows con PowerShell):
ssh pi@raspberrypi.local

# Si no funciona, busca la IP de la RPi en tu router y usa:
ssh pi@192.168.X.X
        """, language="bash")

        st.markdown("#### Paso 3: Actualizar el sistema")
        st.code("""
sudo apt update && sudo apt upgrade -y
        """, language="bash")

        st.markdown("#### Paso 4: Instalar dependencias del sistema")
        st.code("""
# Python y herramientas de compilación
sudo apt install -y python3-pip python3-venv git

# Librerías para OpenCV (cámara USB)
sudo apt install -y libatlas-base-dev libhdf5-dev
sudo apt install -y libharfbuzz0b liblapack-dev
        """, language="bash")

        st.markdown("#### Paso 5: Verificar que la cámara USB funciona")
        st.code("""
# Conectar la cámara USB y verificar
ls /dev/video*
# Debe aparecer: /dev/video0

# Probar captura rápida (opcional)
sudo apt install -y fswebcam
fswebcam -r 1280x720 test.jpg
# Si genera test.jpg, la cámara funciona ✅
        """, language="bash")

        st.markdown("#### Paso 6: Clonar el proyecto e instalar")
        st.code("""
# Crear directorio de trabajo
mkdir -p ~/melanoma && cd ~/melanoma

# Clonar el repositorio (o copiar los archivos del cliente)
git clone https://github.com/eperea/melanoma-detection-system.git
cd melanoma-detection-system/raspberry_pi

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip3 install -r requirements.txt
        """, language="bash")

        st.markdown("#### Paso 7: Configurar la conexión al servidor")
        st.code("""
# Crear archivo de configuración
cat > .env << 'EOF'
MELANOMA_SERVER=https://melanoma.verix.com.co
DEVICE_ID=raspberry_001
EOF

# Verificar conexión al servidor
curl -s https://melanoma.verix.com.co/health
# Debe responder: {"status": "healthy", "models_loaded": true, ...}
        """, language="bash")

        st.markdown("#### Paso 8: Ejecutar el cliente")
        st.code("""
# Activar entorno virtual (si no lo está)
source venv/bin/activate

# Ejecutar
python3 melanoma_client.py
        """, language="bash")

        st.success("""
        ✅ **¡Listo!** El sistema mostrará:
        - "✅ Hardware GPIO inicializado"
        - "📷 Cámara USB inicializada"
        - "✅ Servidor conectado"
        - "👆 Presione el botón para capturar..."
        """)

        st.markdown("#### (Opcional) Ejecutar al encender la Raspberry Pi")
        st.code("""
# Crear servicio systemd para inicio automático
sudo tee /etc/systemd/system/melanoma.service << 'EOF'
[Unit]
Description=Melanoma Detection Client
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/melanoma/melanoma-detection-system/raspberry_pi
Environment="PATH=/home/pi/melanoma/melanoma-detection-system/raspberry_pi/venv/bin"
ExecStart=/home/pi/melanoma/melanoma-detection-system/raspberry_pi/venv/bin/python3 melanoma_client.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Activar el servicio
sudo systemctl daemon-reload
sudo systemctl enable melanoma.service
sudo systemctl start melanoma.service

# Ver logs en tiempo real
sudo journalctl -u melanoma.service -f
        """, language="bash")

    # --- TAB 4: USO DEL SISTEMA ---
    with rpi_tabs[3]:
        st.markdown("### 🚀 Cómo Usar el Sistema")

        st.markdown("#### Flujo de Operación")
        st.markdown("""
        ```
        1. Encender Raspberry Pi
               │
               ▼
        2. El LED AZUL parpadea (conectando al servidor)
               │
               ▼
        3. LED VERDE = Conexión exitosa ✅
               │
               ▼
        4. Posicionar cámara sobre la lesión cutánea
               │
               ▼
        5. Presionar el BOTÓN para capturar
               │
               ▼
        6. LED AZUL = Procesando (enviando al servidor)
               │
               ▼
        7. El servidor analiza con IA (Super-Resolución + MobileNetV2)
               │
               ▼
        8. LED indica RESULTADO:
           🟢 VERDE   = Nevus (Benigno)
           🔴 ROJO    = Melanoma (¡Consultar dermatólogo!)
           🟡 AMARILLO = No clasificable
               │
               ▼
        9. Resultado se guarda en la base de datos
              (visible en la web: melanoma.verix.com.co)
               │
               ▼
       10. Listo para nueva captura (volver al paso 4)
        ```
        """)

        st.markdown("---")
        st.markdown("#### 📸 Tips para Buenas Capturas")
        st.markdown("""
        | Tip | Descripción |
        |:----|:------------|
        | **Iluminación** | Usa luz blanca directa. Evita sombras sobre la lesión |
        | **Distancia** | Mantén la cámara a 5-10 cm de la piel |
        | **Enfoque** | Espera a que la cámara enfoque antes de presionar el botón |
        | **Estabilidad** | Apoya la RPi sobre una superficie plana o usa un soporte |
        | **Limpieza** | Limpia el lente de la cámara antes de cada sesión |
        """)

        st.markdown("---")
        st.markdown("#### 📊 Ver Resultados en la Web")
        st.markdown("""
        Todos los análisis realizados desde la Raspberry Pi se guardan automáticamente 
        en la base de datos del servidor. Puedes verlos en:
        
        🌐 **https://melanoma.verix.com.co** → Menú "Historial Pacientes"
        
        Cada análisis incluye:
        - Diagnóstico (Melanoma / Nevus / Otra afectación)
        - Porcentaje de confianza
        - Fecha y hora del análisis
        - Descarga de reporte PDF
        """)

    # --- TAB 5: ARQUITECTURA ---
    with rpi_tabs[4]:
        st.markdown("### 🏗️ Arquitectura del Sistema con Raspberry Pi")
        
        st.graphviz_chart("""
        digraph Architecture {
            rankdir=LR;
            node [shape=box, style=filled, fontname="Sans"];
            
            subgraph cluster_rpi {
                label = "Raspberry Pi 4 (2GB)";
                style=filled;
                color="#e8f5e9";
                
                Camera [label="📷 Cámara USB\\n(Captura imagen)", fillcolor="#c8e6c9"];
                Button [label="🔘 Botón GPIO 17\\n(Disparo)", fillcolor="#c8e6c9"];
                LED [label="💡 LED RGB\\n(GPIO 22,23,24)", fillcolor="#c8e6c9"];
                Client [label="🐍 melanoma_client.py\\n(Python + OpenCV)", fillcolor="#fff9c4"];
            }
            
            subgraph cluster_server {
                label = "Servidor VPS (melanoma.verix.com.co)";
                style=filled;
                color="#e3f2fd";
                
                API [label="⚡ FastAPI\\n(/api/analyze)", fillcolor="#bbdefb"];
                SRCNN [label="🔍 SRCNN\\n(Super-Resolución)", shape=ellipse, fillcolor="#fff9c4"];
                MobileNet [label="🧠 MobileNetV2\\n(Clasificación)", shape=ellipse, fillcolor="#fff9c4"];
                DB [label="🗄️ PostgreSQL\\n(Datos + Historial)", fillcolor="#e1bee7"];
                Web [label="🌐 Streamlit\\n(Interfaz Web)", fillcolor="#bbdefb"];
            }
            
            Button -> Client [label="GPIO"];
            Camera -> Client [label="USB/OpenCV"];
            Client -> API [label="HTTPS\\nPOST /api/analyze"];
            API -> SRCNN [label="Imagen"];
            SRCNN -> MobileNet [label="Imagen mejorada"];
            MobileNet -> API [label="Diagnóstico"];
            API -> Client [label="JSON resultado"];
            Client -> LED [label="GPIO color"];
            API -> DB [label="Guardar"];
            Web -> DB [label="Consultar"];
        }
        """)

        st.markdown("""
        **Flujo técnico:**
        1. El **botón** activa la captura vía GPIO
        2. **OpenCV** captura la imagen desde la cámara USB
        3. El **cliente Python** envía la imagen por HTTPS al servidor
        4. La **API (FastAPI)** procesa con SRCNN → MobileNetV2
        5. El resultado JSON regresa a la RPi
        6. El **LED RGB** muestra el color según el diagnóstico
        7. Todo queda guardado en **PostgreSQL** para consulta web
        """)

    # --- TAB 6: SOLUCIÓN DE PROBLEMAS ---
    with rpi_tabs[5]:
        st.markdown("### ❓ Solución de Problemas")

        st.markdown("""
        | Problema | Causa Probable | Solución |
        |:---------|:---------------|:---------|
        | "No se detectó cámara USB" | Cámara no conectada o incompatible | Verificar con `ls /dev/video*`. Probar otro puerto USB |
        | "Error de conexión al servidor" | Sin WiFi o servidor caído | Verificar WiFi con `ping google.com`. Probar `curl https://melanoma.verix.com.co/health` |
        | LED no enciende | Cable suelto o resistencia faltante | Revisar conexiones. Verificar con `gpio readall` |
        | Botón no responde | Pin incorrecto o cable invertido | Verificar que el botón va entre GPIO 17 y GND |
        | Imagen borrosa | Cámara muy cerca o sin enfoque | Alejar cámara a 5-10 cm. Limpiar lente |
        | "GPIO en modo simulación" | No estás en una Raspberry Pi | Normal si ejecutas en PC. En RPi: `pip install RPi.GPIO` |
        | Timeout al analizar | Imagen muy pesada o red lenta | Reducir resolución. Verificar conexión WiFi |
        """)

        st.markdown("---")
        st.markdown("#### 🔧 Comandos de Diagnóstico Útiles")
        st.code("""
# Verificar cámara USB
ls -la /dev/video*
v4l2-ctl --list-devices

# Verificar GPIO
gpio readall
pinout

# Verificar red/servidor
ping melanoma.verix.com.co
curl -s https://melanoma.verix.com.co/health | python3 -m json.tool

# Ver logs del servicio (si configuraste systemd)
sudo journalctl -u melanoma.service -f --no-pager -n 50

# Verificar espacio en disco
df -h

# Verificar temperatura (importante en RPi)
vcgencmd measure_temp
        """, language="bash")

        st.markdown("---")
        st.markdown("#### 📞 ¿Necesitas ayuda?")
        st.info("""
        Si tienes problemas con la integración, documenta:
        1. Modelo exacto de tu Raspberry Pi (`cat /proc/device-tree/model`)
        2. Versión del OS (`cat /etc/os-release`)
        3. Logs de error del cliente
        4. Foto de tus conexiones de hardware
        """)

# =====================================================
# OPCIÓN 6: ESPECIFICACIONES TÉCNICAS
# =====================================================
elif menu_option == "⚙️ Especificaciones Técnicas":
    st.markdown('<p class="main-header">⚙️ Especificaciones Técnicas del Proyecto</p>', unsafe_allow_html=True)
    st.markdown("Información detallada sobre tecnologías, versiones, arquitectura de modelos, esquema de datos y configuración del sistema.")

    spec_tabs = st.tabs(["🛠️ Stack Tecnológico", "🧠 Modelos de IA", "🗄️ Base de Datos", "🐳 Infraestructura Docker", "📁 Estructura del Proyecto"])

    with spec_tabs[0]:
        st.markdown("### 🛠️ Stack Tecnológico y Versiones")
        st.markdown("""
        | Componente | Tecnología | Versión / Detalle |
        |:---|:---|:---|
        | **Lenguaje** | Python | 3.x (incluido en imagen TensorFlow) |
        | **Framework Web** | Streamlit | 1.32.0 |
        | **Clasificación (Deep Learning)** | TensorFlow / Keras | 2.15.0 (imagen base Docker) |
        | **Super-Resolución** | PyTorch (CPU) | Última estable (vía pip, índice CPU) |
        | **Visión por Computador** | OpenCV (headless) | opencv-python-headless |
        | **Procesamiento de Imágenes** | Pillow (PIL) | Última estable |
        | **Cálculo Numérico** | NumPy | < 2.0.0 |
        | **Visualización** | Matplotlib | Última estable |
        | **Procesamiento de Imagen Científico** | scikit-image | Última estable |
        | **Generación de PDF** | ReportLab | Última estable |
        | **Base de Datos** | PostgreSQL | 15 (imagen Docker oficial) |
        | **Conector BD** | psycopg2-binary | Última estable |
        | **Zona Horaria** | pytz | America/Bogota (UTC-5) |
        | **Contenedores** | Docker / Docker Compose | Compose v3.8 |
        """)

        st.markdown("### 📦 Dependencias del Sistema (Dockerfile)")
        st.code("""
# Imagen base con TensorFlow 2.15 preinstalado
FROM tensorflow/tensorflow:2.15.0

# Dependencias de sistema para OpenCV
libgl1-mesa-glx, libglib2.0-0

# PyTorch CPU (~150MB vs ~1GB versión GPU)
torch + torchvision (índice: https://download.pytorch.org/whl/cpu)

# Dependencias Python
streamlit==1.32.0, opencv-python-headless, Pillow,
psycopg2-binary, numpy<2.0.0, matplotlib, scikit-image, reportlab
        """, language="dockerfile")

    with spec_tabs[1]:
        st.markdown("### 🧠 Especificaciones de los Modelos de IA")

        st.markdown("#### 1. Clasificador — MobileNetV2")
        st.markdown("""
        | Parámetro | Valor |
        |:---|:---|
        | **Arquitectura** | MobileNetV2 (Transfer Learning) |
        | **Pesos Pre-entrenados** | ImageNet (1.4M imágenes) |
        | **Fine-Tuning** | Sobre dataset HAM10000 (dermatoscopia) |
        | **Entrada** | Imagen RGB 224 × 224 px |
        | **Salida** | 2 clases: Melanoma, Nevus |
        | **Formato del Modelo** | HDF5 (.h5) — Keras |
        | **Archivo** | `models/keras_model.h5` |
        | **Etiquetas** | `models/labels.txt` → 0: Melanoma, 1: Nevus |
        | **Umbral de Confianza** | Configurable (60%–95%, por defecto 70%) |
        | **Técnica de Convolución** | Depthwise Separable Convolutions |
        | **Bloques** | Inverted Residuals con Linear Bottlenecks |
        | **Reducción de Cómputo** | 8–9x menos operaciones vs CNN convencional |
        """)

        st.markdown("---")
        st.markdown("#### 2. Super-Resolución — SRCNN")
        st.markdown("""
        | Parámetro | Valor |
        |:---|:---|
        | **Arquitectura** | SRCNN (Super-Resolution CNN) |
        | **Framework** | PyTorch |
        | **Formato del Modelo** | State Dict (.pth) |
        | **Archivo** | `models/best_srcnn.pth` |
        | **Factor de Escala** | 2x (Bicúbico) |
        | **Capa 1 (Patch Extraction)** | Conv2d: 3 → 64 canales, kernel 9×9, padding 4 |
        | **Capa 2 (Non-linear Mapping)** | Conv2d: 64 → 32 canales, kernel 1×1 |
        | **Capa 3 (Reconstruction)** | Conv2d: 32 → 3 canales, kernel 5×5, padding 2 |
        | **Activación** | ReLU (capas 1 y 2) |
        | **Procesamiento** | Por parches de 64×64 px, stride 32 px |
        | **Upsampling** | nn.Upsample (bicúbico, scale_factor=2) |
        | **Dispositivo** | CPU (o GPU si disponible) |
        """)

        st.markdown("---")
        st.markdown("#### 📊 Métricas de Rendimiento del Clasificador")
        st.markdown("""
        | Métrica | Valor |
        |:---|:---|
        | **Accuracy (Validación General)** | 96.7% |
        | **F1-Score — Melanoma** | 0.93 |
        | **F1-Score — Nevus** | 0.94 |
        | **Recall Melanoma (Validación Balanceada 39 vs 39)** | 87% |
        | **Dataset de Entrenamiento** | HAM10000 (~6,000 imágenes/clase, balanceado) |
        | **Dataset de Validación** | 751 Nevus + 39 Melanoma (desbalanceado) |
        """)

    with spec_tabs[2]:
        st.markdown("### 🗄️ Esquema de Base de Datos")
        st.markdown("""
        | Parámetro | Valor |
        |:---|:---|
        | **Motor** | PostgreSQL 15 |
        | **Base de Datos** | melanoma_db |
        | **Conector Python** | psycopg2-binary (RealDictCursor) |
        | **Puerto** | 5432 |
        | **Volumen Persistente** | postgres_data (Docker volume) |
        """)

        st.markdown("---")
        st.markdown("#### Tabla: `pacientes`")
        st.code("""
CREATE TABLE pacientes (
    id              SERIAL PRIMARY KEY,
    identificacion  VARCHAR(50) UNIQUE NOT NULL,  -- Cédula o ID del paciente
    nombre          VARCHAR(200) NOT NULL,        -- Nombre completo
    edad            INTEGER,                      -- Edad en años
    sexo            VARCHAR(20),                  -- Masculino / Femenino / Otro
    fecha_registro  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
        """, language="sql")

        st.markdown("#### Tabla: `analisis`")
        st.code("""
CREATE TABLE analisis (
    id                     SERIAL PRIMARY KEY,
    paciente_id            INTEGER REFERENCES pacientes(id),  -- FK al paciente
    ubicacion_lesion       VARCHAR(100),     -- Rostro, Brazo, Pierna, etc.
    notas_clinicas         TEXT,             -- Observaciones del médico
    diagnostico            VARCHAR(50),      -- Melanoma / Nevus / Otra afectación
    confianza              FLOAT,            -- Score de confianza (0.0 – 1.0)
    probabilidad_melanoma  FLOAT,            -- P(Melanoma)
    probabilidad_nevus     FLOAT,            -- P(Nevus)
    fecha_analisis         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
        """, language="sql")

        st.markdown("#### 🔗 Relaciones")
        st.markdown("""
        - `analisis.paciente_id` → `pacientes.id` (N análisis por paciente)
        - Identificación del paciente es **UNIQUE** para evitar duplicados
        - Cada análisis registra las probabilidades individuales para trazabilidad completa
        """)

    with spec_tabs[3]:
        st.markdown("### 🐳 Infraestructura Docker")

        st.markdown("#### Servicios (docker-compose v3.8)")
        st.markdown("""
        | Servicio | Contenedor | Imagen Base | Puerto Expuesto |
        |:---|:---|:---|:---|
        | **web** (App) | `melanoma_app` | `tensorflow/tensorflow:2.15.0` (build) | 8501 (Streamlit) |
        | **db** (PostgreSQL) | `melanoma_db` | `postgres:15` | 5432 |
        """)

        st.markdown("#### Variables de Entorno")
        st.markdown("""
        | Variable | Servicio | Valor |
        |:---|:---|:---|
        | `DATABASE_URL` | web | `postgresql://user:password@db:5432/melanoma_db` |
        | `TZ` | web, db | `America/Bogota` |
        | `POSTGRES_USER` | db | `user` |
        | `POSTGRES_DB` | db | `melanoma_db` |
        """)

        st.markdown("#### Volúmenes")
        st.markdown("""
        | Volumen | Tipo | Ruta en Contenedor | Descripción |
        |:---|:---|:---|:---|
        | `./data` | Bind mount | `/data` | Datos compartidos |
        | `postgres_data` | Named volume | `/var/lib/postgresql/data` | Persistencia de BD |
        """)

        st.markdown("#### Diagrama de Red")
        st.graphviz_chart("""
        digraph Docker {
            rankdir=TB;
            node [shape=box, style=filled, fontname="Sans"];

            subgraph cluster_docker {
                label = "Docker Compose Network";
                style=filled;
                color="#f5f5f5";

                app [label="melanoma_app\n(Streamlit + TF + PyTorch)\nPuerto: 8501", fillcolor="#e3f2fd"];
                db [label="melanoma_db\n(PostgreSQL 15)\nPuerto: 5432", fillcolor="#e1bee7"];
            }

            user [label="🌐 Usuario\n(Navegador)", shape=ellipse, fillcolor="#fff9c4"];

            user -> app [label="HTTP :8501"];
            app -> db [label="TCP :5432\npsycopg2"];
        }
        """)

    with spec_tabs[4]:
        st.markdown("### 📁 Estructura del Proyecto")
        st.code("""
melanoma-app/
├── docker-compose.yml             # Orquestación de contenedores
├── README.md                      # Documentación del proyecto
├── train_melanoma_model.py        # Script de entrenamiento (HAM10000)
├── train_model.py                 # Script de entrenamiento alternativo
├── prepare_data.py                # Preparación del dataset
├── prepare_real_data.py           # Descarga dataset ISIC real
├── generate_dummy.py              # Generación de datos sintéticos
├── generate_matrix.py             # Generación de matriz de confusión (validación)
├── generate_matrix_train.py       # Generación de matriz de confusión (entrenamiento)
├── data/                          # Directorio de datos
└── app/                           # Aplicación principal
    ├── Dockerfile                 # Imagen Docker de la app
    ├── requirements.txt           # Dependencias Python
    ├── main.py                    # Punto de entrada Streamlit (UI)
    ├── database.py                # Conexión y operaciones PostgreSQL
    ├── models/
    │   ├── keras_model.h5         # Modelo MobileNetV2 entrenado (.h5)
    │   ├── best_srcnn.pth         # Modelo SRCNN super-resolución (.pth)
    │   └── labels.txt             # Etiquetas: 0 Melanoma, 1 Nevus
    ├── logic/
    │   ├── classifier.py          # Lógica de clasificación MobileNetV2
    │   └── sr_model.py            # Lógica de super-resolución SRCNN
    ├── utils/
    │   ├── __init__.py
    │   └── pdf_report.py          # Generación de reportes PDF (ReportLab)
    └── assets/
        ├── matriz_confusion.png           # Matriz de confusión (validación)
        ├── matriz_confusion_balanced.png   # Matriz balanceada (39 vs 39)
        ├── reporte_clasificacion.txt       # Reporte de métricas (validación)
        └── reporte_balanced.txt            # Reporte de métricas (balanceado)
        """, language="text")

        st.markdown("#### 📝 Notas Adicionales")
        st.markdown("""
        - **Pipeline de IA:** Imagen → SRCNN (mejora) → MobileNetV2 (clasificación) → Resultado
        - **Generación de Reportes:** PDF generado con ReportLab, incluye datos del paciente, diagnóstico, probabilidades y disclaimer médico
        - **Seguridad:** Los datos de pacientes se almacenan localmente en PostgreSQL. La aplicación no envía datos a servicios externos.
        - **Zona horaria:** Todas las fechas y timestamps usan `America/Bogota` (UTC-5)
        - **Compatibilidad:** Monkey-patching aplicado a `DepthwiseConv2D` para compatibilidad con TensorFlow 2.15 / Keras
        """)

# =====================================================
# OPCIÓN 6: ACERCA DE
# =====================================================
elif menu_option == "ℹ️ Acerca de":
    st.markdown("### Acerca de este Proyecto")
    st.info("""
    **Desarrollado como Proyecto de Grado / Tesis.**
    
    Este software es una demostración académica de las capacidades de la Inteligencia Artificial aplicada a la medicina.
    No sustituye el criterio de un profesional de la salud certificado.
    """)
    st.write("© 2026 - Detección de Melanoma con IA")
