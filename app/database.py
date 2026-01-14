import os
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://user:password@localhost:5432/melanoma_db")

def get_connection():
    """Obtiene conexión a la base de datos PostgreSQL."""
    return psycopg2.connect(DATABASE_URL)

def init_database():
    """Inicializa las tablas necesarias."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Tabla de pacientes
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pacientes (
                id SERIAL PRIMARY KEY,
                identificacion VARCHAR(50) UNIQUE NOT NULL,
                nombre VARCHAR(200) NOT NULL,
                edad INTEGER,
                sexo VARCHAR(20),
                fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tabla de análisis
        cur.execute("""
            CREATE TABLE IF NOT EXISTS analisis (
                id SERIAL PRIMARY KEY,
                paciente_id INTEGER REFERENCES pacientes(id),
                ubicacion_lesion VARCHAR(100),
                notas_clinicas TEXT,
                diagnostico VARCHAR(50),
                confianza FLOAT,
                probabilidad_melanoma FLOAT,
                probabilidad_nevus FLOAT,
                fecha_analisis TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        # Tables may already exist, that's fine
        pass

def registrar_paciente(identificacion, nombre, edad, sexo):
    """Registra un nuevo paciente o retorna el existente."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    # Verificar si ya existe
    cur.execute("SELECT * FROM pacientes WHERE identificacion = %s", (identificacion,))
    paciente = cur.fetchone()
    
    if paciente:
        cur.close()
        conn.close()
        return dict(paciente)
    
    # Crear nuevo
    cur.execute("""
        INSERT INTO pacientes (identificacion, nombre, edad, sexo)
        VALUES (%s, %s, %s, %s)
        RETURNING *
    """, (identificacion, nombre, edad, sexo))
    
    paciente = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return dict(paciente)

def guardar_analisis(paciente_id, ubicacion, notas, diagnostico, confianza, prob_melanoma, prob_nevus):
    """Guarda un análisis en la base de datos."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    cur.execute("""
        INSERT INTO analisis (paciente_id, ubicacion_lesion, notas_clinicas, 
                              diagnostico, confianza, probabilidad_melanoma, probabilidad_nevus)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING *
    """, (paciente_id, ubicacion, notas, diagnostico, confianza, prob_melanoma, prob_nevus))
    
    analisis = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return dict(analisis)

def obtener_historial_paciente(identificacion):
    """Obtiene el historial de análisis de un paciente."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    cur.execute("""
        SELECT a.*, p.nombre, p.identificacion
        FROM analisis a
        JOIN pacientes p ON a.paciente_id = p.id
        WHERE p.identificacion = %s
        ORDER BY a.fecha_analisis DESC
    """, (identificacion,))
    
    historial = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(h) for h in historial]

def buscar_paciente(identificacion):
    """Busca un paciente por su identificación."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    cur.execute("SELECT * FROM pacientes WHERE identificacion = %s", (identificacion,))
    paciente = cur.fetchone()
    
    cur.close()
    conn.close()
    return dict(paciente) if paciente else None
