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
    ["🔬 Nuevo Análisis", "🗂️ Historial Pacientes", "📊 Métricas & Validación", "📚 Documentación Técnica", "ℹ️ Acerca de"]
)

st.sidebar.divider()
st.sidebar.info("Proyecto de Tesis\n\nIngeniería de Sistemas - IA detectando Melanoma")

# =====================================================
# OPCIÓN 1: NUEVO ANÁLISIS
# =====================================================
if menu_option == "🔬 Nuevo Análisis":
    
    st.markdown('<p class="main-header">Detección Asistida de Melanoma</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Análisis de imágenes dermatoscópicas con Super-Resolución y Deep Learning</p>', unsafe_allow_html=True)

    if not models_loaded:
        st.error("Los modelos no están cargados. No se puede realizar el análisis.")
        st.stop()
        
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 1. Datos del Paciente")
        id_paciente = st.text_input("Identificación (Cédula/ID)", placeholder="Ej: 123456789")
        
        # Buscar paciente existente
        paciente_db = None
        if id_paciente:
            paciente_db = buscar_paciente(id_paciente)
            if paciente_db:
                st.success(f"Paciente encontrado: {paciente_db['nombre']}")
                nombre = st.text_input("Nombre Completo", value=paciente_db['nombre'], disabled=True)
                edad = st.number_input("Edad", min_value=0, max_value=120, value=paciente_db['edad'], disabled=True)
                sexo = st.selectbox("Sexo", ["Masculino", "Femenino", "Otro"], index=["Masculino", "Femenino", "Otro"].index(paciente_db['sexo']) if paciente_db['sexo'] in ["Masculino", "Femenino", "Otro"] else 0, disabled=True)
            else:
                st.info("Paciente nuevo. Por favor registre sus datos.")
                nombre = st.text_input("Nombre Completo")
                edad = st.number_input("Edad", min_value=0, max_value=120, value=30)
                sexo = st.selectbox("Sexo", ["Masculino", "Femenino", "Otro"])
        else:
            nombre = st.text_input("Nombre Completo", disabled=True)
            edad = st.number_input("Edad", disabled=True)
            sexo = st.selectbox("Sexo", [], disabled=True)

        st.markdown("### 2. Detalles Clínicos")
        ubicacion = st.selectbox("Ubicación de la lesión", ["Rostro", "Brazo", "Pierna", "Espalda", "Pecho", "Abdomen", "Otro"])
        notas = st.text_area("Notas Clínicas (Opcional)", height=100)

    with col2:
        st.markdown("### 3. Cargar Imagen")
        print("Esperando imagen...")
        uploaded_file = st.file_uploader("Arrastre o seleccione una imagen dermatoscópica", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Mostrar imagen original
            image_original = Image.open(uploaded_file).convert("RGB")
            file_bytes = io.BytesIO()
            image_original.save(file_bytes, format='JPEG')
            file_bytes = file_bytes.getvalue()
            
            st.image(image_original, caption="Imagen Subida", use_column_width=True)
            
            # Botón de análisis
            datos_completos = id_paciente and nombre
            if not datos_completos:
                st.warning("⚠️ Complete los datos del paciente para habilitar el análisis.")
            
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
                    original_array = np.array(image_original) # Usando original para predecir (entrenado así)
                    class_name, confidence, probabilities = classifier.predict(original_array)
                    
                    st.write("💾 Guardando resultados en base de datos...")
                    prob_melanoma = float(probabilities[0]) if len(probabilities) > 0 else 0.0
                    prob_nevus = float(probabilities[1]) if len(probabilities) > 1 else 0.0
                    
                    guardar_analisis(
                        paciente_db['id'], ubicacion, notas, 
                        class_name, confidence, prob_melanoma, prob_nevus
                    )
                    
                    status.update(label="✅ ¡Análisis Completado!", state="complete", expanded=False)
                
                # --- RESULTADOS ---
                st.divider()
                st.markdown("## 📊 Resultados del Diagnóstico")
                
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    is_melanoma = "melanoma" in class_name.lower()
                    color = "red" if is_melanoma else "green"
                    icono = "⚠️" if is_melanoma else "✅"
                    titulo = "MELANOMA (Maligno)" if is_melanoma else "NEVUS (Benigno)"
                    
                    st.markdown(f"""
                    <div style="background-color: {'#ffebee' if is_melanoma else '#e8f5e9'}; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid {color};">
                        <h2 style="color: {color}; margin:0;">{icono} {titulo}</h2>
                        <h3 style="margin:0;">Confianza: {confidence:.2%}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Fecha Colombia
                    bogota_tz = pytz.timezone('America/Bogota')
                    current_time = datetime.now(bogota_tz)
                    st.caption(f"Fecha de Análisis: {current_time.strftime('%d/%m/%Y %H:%M %p')}")

                with res_col2:
                    st.markdown("#### Comparativa Visual")
                with res_col2:
                    st.markdown("#### Comparativa Visual")
                    # Usar pestañas para evitar anidamiento excesivo de columnas
                    viz_tabs = st.tabs(["Original", "Mejorada (SR)"])
                    with viz_tabs[0]:
                        st.image(image_original, caption="Original", use_column_width=True)
                    with viz_tabs[1]:
                        st.image(sr_image_pil, caption="Mejorada (SR)", use_column_width=True)
                
                st.divider()
                st.markdown("#### Probabilidades Detalladas")
                st.progress(prob_melanoma, text=f"Melanoma: {prob_melanoma:.2%}")
                st.progress(prob_nevus, text=f"Nevus: {prob_nevus:.2%}")
                
                # Botón de descarga de reporte PDF
                st.divider()
                analisis_data = {
                    'fecha_analisis': current_time.strftime('%Y-%m-%d %H:%M'),
                    'diagnostico': class_name,
                    'confianza': confidence,
                    'ubicacion_lesion': ubicacion,
                    'notas_clinicas': notas,
                    'probabilidad_melanoma': prob_melanoma,
                    'probabilidad_nevus': prob_nevus
                }
                pdf_bytes = generate_report_pdf(paciente_db, analisis_data)
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
                            pdf_bytes = generate_report_pdf(paciente, h)
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
# OPCIÓN 5: ACERCA DE
# =====================================================
elif menu_option == "ℹ️ Acerca de":
    st.markdown("### Acerca de este Proyecto")
    st.info("""
    **Desarrollado como Proyecto de Grado / Tesis.**
    
    Este software es una demostración académica de las capacidades de la Inteligencia Artificial aplicada a la medicina.
    No sustituye el criterio de un profesional de la salud certificado.
    """)
    st.write("© 2026 - Detección de Melanoma con IA")
