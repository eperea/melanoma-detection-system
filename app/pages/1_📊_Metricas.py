import streamlit as st
import os

st.set_page_config(page_title="Métricas del Modelo", page_icon="📊", layout="wide")

st.title("📊 Métricas de Evaluación del Modelo")
st.markdown("""
Esta sección presenta el rendimiento del modelo de clasificación (MobileNetV2) evaluado con el conjunto de validación.
""")

# Rutas de archivos (asumiendo que están en assets relativo a la raíz de ejecución app/)
# En Docker, workdir es /app, así que assets está en /app/assets
IMG_PATH = "assets/matriz_confusion.png"
REPORT_PATH = "assets/reporte_clasificacion.txt"

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Matriz de Confusión")
    if os.path.exists(IMG_PATH):
        st.image(IMG_PATH, caption="Matriz de Confusión (Validación)", use_column_width=True)
    else:
        st.error(f"No se encontró la imagen en {IMG_PATH}")
        st.info("Asegúrese de haber ejecutado generate_matrix.py y subido los assets.")

with col2:
    st.subheader("Reporte de Clasificación")
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, "r") as f:
            report_text = f.read()
        st.text(report_text)
        
        st.markdown("""
        **Interpretación:**
        - **Precision (Precisión):** De todas las lesiones que el modelo predijo como Melanoma, ¿cuántas eran realmente Melanoma?
        - **Recall (Sensibilidad):** De todos los Melanomas reales, ¿cuántos detectó el modelo?
        - **F1-Score:** Media armónica entre Precision y Recall.
        """)
    else:
        st.error(f"No se encontró el reporte en {REPORT_PATH}")

st.divider()
st.caption("Nota: Estas métricas corresponden al conjunto de validación (no visto durante el entrenamiento).")
