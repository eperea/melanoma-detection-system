from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, red, green
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime
import io
import os

def generate_report_pdf(
    paciente_nombre,
    paciente_id,
    paciente_edad,
    paciente_sexo,
    ubicacion_lesion,
    notas_clinicas,
    diagnostico,
    confianza,
    prob_melanoma,
    prob_nevus,
    imagen_original_path=None,
    imagen_sr_path=None
):
    """
    Genera un reporte PDF con los resultados del análisis de melanoma.
    Retorna los bytes del PDF para descarga.
    """
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    # Estilos
    styles = getSampleStyleSheet()
    
    # Estilos personalizados
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=HexColor('#1E3A5F'),
        alignment=TA_CENTER,
        spaceAfter=20
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=HexColor('#333333'),
        spaceBefore=15,
        spaceAfter=10
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        leading=14
    )
    
    alert_melanoma_style = ParagraphStyle(
        'AlertMelanoma',
        parent=styles['Normal'],
        fontSize=12,
        textColor=red,
        backColor=HexColor('#FFEBEE'),
        borderPadding=10,
        leading=16
    )
    
    alert_nevus_style = ParagraphStyle(
        'AlertNevus',
        parent=styles['Normal'],
        fontSize=12,
        textColor=green,
        backColor=HexColor('#E8F5E9'),
        borderPadding=10,
        leading=16
    )
    
    # Contenido del PDF
    elements = []
    
    # Título
    elements.append(Paragraph("🔬 Reporte de Análisis Dermatoscópico", title_style))
    elements.append(Paragraph("Sistema de Detección de Melanoma con Super-Resolución", styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Fecha y hora
    fecha_actual = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    elements.append(Paragraph(f"<b>Fecha de Análisis:</b> {fecha_actual}", normal_style))
    elements.append(Spacer(1, 15))
    
    # Datos del Paciente
    elements.append(Paragraph("📋 Datos del Paciente", subtitle_style))
    
    datos_paciente = [
        ["Nombre:", paciente_nombre],
        ["Identificación:", paciente_id],
        ["Edad:", f"{paciente_edad} años"],
        ["Sexo:", paciente_sexo],
        ["Ubicación de Lesión:", ubicacion_lesion]
    ]
    
    tabla_paciente = Table(datos_paciente, colWidths=[2*inch, 4*inch])
    tabla_paciente.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), HexColor('#F5F5F5')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#DDDDDD')),
    ]))
    elements.append(tabla_paciente)
    elements.append(Spacer(1, 15))
    
    # Notas clínicas
    if notas_clinicas:
        elements.append(Paragraph("<b>Notas Clínicas:</b>", normal_style))
        elements.append(Paragraph(notas_clinicas, normal_style))
        elements.append(Spacer(1, 15))
    
    # Resultados del Diagnóstico
    elements.append(Paragraph("📊 Resultados del Diagnóstico", subtitle_style))
    
    is_melanoma = "melanoma" in diagnostico.lower()
    
    resultados = [
        ["Diagnóstico:", diagnostico.upper()],
        ["Confianza:", f"{confianza:.1%}"],
        ["P(Melanoma):", f"{prob_melanoma:.2%}"],
        ["P(Nevus):", f"{prob_nevus:.2%}"]
    ]
    
    tabla_resultados = Table(resultados, colWidths=[2*inch, 4*inch])
    
    # Color según resultado
    color_diagnostico = red if is_melanoma else green
    
    tabla_resultados.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), HexColor('#F5F5F5')),
        ('BACKGROUND', (1, 0), (1, 0), HexColor('#FFEBEE') if is_melanoma else HexColor('#E8F5E9')),
        ('TEXTCOLOR', (1, 0), (1, 0), color_diagnostico),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#DDDDDD')),
    ]))
    elements.append(tabla_resultados)
    elements.append(Spacer(1, 20))
    
    # Recomendación clínica
    elements.append(Paragraph("⚕️ Recomendación Clínica", subtitle_style))
    
    if is_melanoma:
        recomendacion = """
        <b>⚠️ ALERTA - Posible Melanoma Detectado</b><br/><br/>
        Se han detectado patrones compatibles con melanoma. Se recomienda 
        <b>derivación inmediata a dermatología oncológica</b> para biopsia 
        y evaluación especializada.<br/><br/>
        Este resultado es una herramienta de apoyo diagnóstico y no reemplaza 
        el criterio médico profesional.
        """
        elements.append(Paragraph(recomendacion, alert_melanoma_style))
    else:
        recomendacion = """
        <b>✅ Lesión Benigna - Nevus</b><br/><br/>
        La lesión presenta características de un nevus benigno. Se sugiere 
        <b>monitoreo periódico</b> y revisión si hay cambios en tamaño, forma o color.<br/><br/>
        Este resultado es una herramienta de apoyo diagnóstico y no reemplaza 
        el criterio médico profesional.
        """
        elements.append(Paragraph(recomendacion, alert_nevus_style))
    
    elements.append(Spacer(1, 30))
    
    # Disclaimer
    elements.append(Paragraph("─" * 60, styles['Normal']))
    disclaimer = """
    <i><font size="9">
    Este reporte fue generado automáticamente por el Sistema de Detección de Melanoma 
    utilizando técnicas de Inteligencia Artificial (Super-Resolución SRCNN + Clasificación MobileNetV2).
    Los resultados son orientativos y deben ser validados por un profesional médico especializado.
    </font></i>
    """
    elements.append(Paragraph(disclaimer, styles['Normal']))
    
    # Generar PDF
    doc.build(elements)
    
    buffer.seek(0)
    return buffer.getvalue()
