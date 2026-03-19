"""
Cliente Melanoma para Raspberry Pi — Versión Económica
======================================================
Sistema de captura de imágenes dermatoscópicas con análisis remoto por IA

VERSIÓN ECONÓMICA (~$80 USD / ~$320.000 COP):
- Raspberry Pi 4 Model B (2GB RAM)
- Cámara USB genérica (compatible con OpenCV)
- Botón pulsador de captura (GPIO 17)
- LED RGB para indicación de resultado (GPIO 22, 23, 24)
- Cables jumper, resistencias, MicroSD 32GB, fuente 5V/3A

Autor: Sistema de Detección de Melanoma
Versión: 2.0.0 (Económica - USB Camera)
"""

import os
import sys
import time
import requests
import io
import json
from datetime import datetime
from PIL import Image

# Configuración del servidor
SERVER_URL = os.environ.get("MELANOMA_SERVER", "https://melanoma.verix.com.co")
DEVICE_ID = os.environ.get("DEVICE_ID", "raspberry_001")

# Intentar importar módulos de Raspberry Pi
RASPBERRY_PI = False
HAS_GPIO = False
HAS_OPENCV = False

try:
    import RPi.GPIO as GPIO
    HAS_GPIO = True
except ImportError:
    pass

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    pass

RASPBERRY_PI = HAS_GPIO  # Estamos en RPi si GPIO está disponible

if not RASPBERRY_PI:
    print("⚠️ Ejecutando en modo simulación (sin hardware Raspberry Pi)")
if not HAS_OPENCV:
    print("⚠️ OpenCV no disponible — cámara USB deshabilitada")


class HardwareController:
    """Controlador de hardware GPIO para Raspberry Pi"""
    
    # Pines GPIO (BCM)
    PIN_BUTTON = 17      # Botón de captura
    PIN_LED_RED = 22     # LED Rojo (Melanoma/Error)
    PIN_LED_GREEN = 23   # LED Verde (Benigno/OK)
    PIN_LED_BLUE = 24    # LED Azul (Procesando)
    
    def __init__(self):
        if HAS_GPIO:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            # Configurar pines
            GPIO.setup(self.PIN_BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.setup(self.PIN_LED_RED, GPIO.OUT)
            GPIO.setup(self.PIN_LED_GREEN, GPIO.OUT)
            GPIO.setup(self.PIN_LED_BLUE, GPIO.OUT)
            
            # Apagar todos los LEDs
            self.leds_off()
            
            print("✅ Hardware GPIO inicializado")
        else:
            print("⚠️ GPIO en modo simulación")
    
    def leds_off(self):
        """Apagar todos los LEDs"""
        if HAS_GPIO:
            GPIO.output(self.PIN_LED_RED, GPIO.LOW)
            GPIO.output(self.PIN_LED_GREEN, GPIO.LOW)
            GPIO.output(self.PIN_LED_BLUE, GPIO.LOW)
    
    def set_led(self, color: str):
        """
        Configurar LED según color
        - RED: Melanoma detectado / Error
        - GREEN: Benigno
        - BLUE: Procesando
        - YELLOW: Resultado inconcluso (rojo + verde)
        """
        self.leds_off()
        
        if HAS_GPIO:
            if color == "RED":
                GPIO.output(self.PIN_LED_RED, GPIO.HIGH)
            elif color == "GREEN":
                GPIO.output(self.PIN_LED_GREEN, GPIO.HIGH)
            elif color == "BLUE":
                GPIO.output(self.PIN_LED_BLUE, GPIO.HIGH)
            elif color == "YELLOW":
                GPIO.output(self.PIN_LED_RED, GPIO.HIGH)
                GPIO.output(self.PIN_LED_GREEN, GPIO.HIGH)
        
        print(f"💡 LED: {color}")
    
    def blink_led(self, color: str, times: int = 3, interval: float = 0.3):
        """Parpadear LED"""
        for _ in range(times):
            self.set_led(color)
            time.sleep(interval)
            self.leds_off()
            time.sleep(interval)
    
    def wait_for_button(self) -> bool:
        """Esperar presión del botón"""
        if HAS_GPIO:
            print("👆 Presione el botón para capturar...")
            GPIO.wait_for_edge(self.PIN_BUTTON, GPIO.FALLING)
            time.sleep(0.1)  # Debounce
            return True
        else:
            input("👆 Presione ENTER para simular captura...")
            return True
    
    def cleanup(self):
        """Limpiar GPIO al salir"""
        if HAS_GPIO:
            GPIO.cleanup()
            print("🔌 GPIO liberado")


class CameraController:
    """
    Controlador de cámara USB para Raspberry Pi (Versión Económica)
    Usa OpenCV para compatibilidad con cualquier cámara USB estándar
    """
    
    def __init__(self, resolution=(1920, 1080), camera_index=0):
        self.resolution = resolution
        self.camera = None
        self.camera_index = camera_index
        
        if HAS_OPENCV:
            try:
                self.camera = cv2.VideoCapture(camera_index)
                if not self.camera.isOpened():
                    raise RuntimeError("No se detectó cámara USB")
                
                # Configurar resolución
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
                
                # Leer resolución real obtenida
                w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                time.sleep(2)  # Warm-up de la cámara
                print(f"📷 Cámara USB inicializada (índice {camera_index}): {w}x{h}")
            except Exception as e:
                print(f"❌ Error inicializando cámara USB: {e}")
                self.camera = None
        else:
            print("📷 Cámara en modo simulación (OpenCV no disponible)")
    
    def capture(self) -> Image.Image:
        """Capturar imagen desde cámara USB"""
        if self.camera and self.camera.isOpened():
            # Descartar frames viejos del buffer
            for _ in range(3):
                self.camera.read()
            
            ret, frame = self.camera.read()
            if ret:
                # OpenCV captura en BGR, convertir a RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                return image
            else:
                print("❌ Error al capturar frame")
                return Image.new('RGB', self.resolution, color='gray')
        else:
            print("🖼️ Generando imagen de prueba (modo simulación)...")
            return Image.new('RGB', self.resolution, color='gray')
    
    def capture_to_file(self, filepath: str):
        """Capturar y guardar en archivo"""
        image = self.capture()
        image.save(filepath, "JPEG", quality=95)
        return filepath
    
    def close(self):
        """Cerrar cámara"""
        if self.camera:
            self.camera.release()
            print("📷 Cámara USB cerrada")


class MelanomaClient:
    """Cliente principal para comunicación con servidor"""
    
    def __init__(self, server_url: str, device_id: str):
        self.server_url = server_url.rstrip('/')
        self.device_id = device_id
        self.session = requests.Session()
        self.session.timeout = 60
    
    def check_connection(self) -> bool:
        """Verificar conexión con servidor"""
        try:
            response = self.session.get(f"{self.server_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Servidor conectado: {data.get('status')}")
                print(f"   Modelos cargados: {data.get('models_loaded')}")
                return True
        except Exception as e:
            print(f"❌ Error de conexión: {e}")
        return False
    
    def analyze_image(
        self,
        image: Image.Image,
        patient_id: str = None,
        location: str = "No especificada",
        notes: str = ""
    ) -> dict:
        """
        Enviar imagen para análisis
        
        Args:
            image: Imagen PIL a analizar
            patient_id: ID del paciente (opcional)
            location: Ubicación de la lesión
            notes: Notas adicionales
            
        Returns:
            dict con resultado del análisis
        """
        # Convertir imagen a bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG', quality=95)
        img_buffer.seek(0)
        
        # Preparar datos
        files = {
            'image': ('capture.jpg', img_buffer, 'image/jpeg')
        }
        data = {
            'device_id': self.device_id,
            'patient_id': patient_id or '',
            'location': location,
            'notes': notes
        }
        
        try:
            response = self.session.post(
                f"{self.server_url}/api/analyze",
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text}"
                }
                
        except requests.exceptions.Timeout:
            return {'success': False, 'error': 'Timeout - servidor no responde'}
        except Exception as e:
            return {'success': False, 'error': str(e)}


def print_result(result: dict):
    """Imprimir resultado de forma legible"""
    print("\n" + "="*50)
    print("📊 RESULTADO DEL ANÁLISIS")
    print("="*50)
    
    if result.get('success'):
        analysis = result.get('analysis', {})
        print(f"🔬 Diagnóstico: {analysis.get('diagnosis', 'N/A')}")
        print(f"📈 Confianza: {analysis.get('confidence', 0)}%")
        print(f"⚠️ Nivel de riesgo: {analysis.get('risk_level', 'N/A')}")
        print(f"💡 Indicador LED: {result.get('led_signal', 'N/A')}")
        print(f"\n📋 Recomendación:")
        print(f"   {result.get('recommendation', 'N/A')}")
        
        if 'probabilities' in analysis:
            print(f"\n📊 Probabilidades:")
            for clase, prob in analysis['probabilities'].items():
                print(f"   - {clase}: {prob}%")
    else:
        print(f"❌ Error: {result.get('error', 'Desconocido')}")
    
    print("="*50 + "\n")


def main():
    """Función principal del cliente"""
    print("\n" + "="*60)
    print("🏥 SISTEMA DE DETECCIÓN DE MELANOMA")
    print("   Módulo Raspberry Pi v1.0")
    print("="*60 + "\n")
    
    # Inicializar componentes
    hardware = HardwareController()
    camera = CameraController()
    client = MelanomaClient(SERVER_URL, DEVICE_ID)
    
    # LED azul: iniciando
    hardware.set_led("BLUE")
    
    # Verificar conexión
    print(f"🌐 Conectando a {SERVER_URL}...")
    if not client.check_connection():
        hardware.blink_led("RED", times=5)
        print("❌ No se pudo conectar al servidor")
        hardware.cleanup()
        camera.close()
        return
    
    hardware.set_led("GREEN")
    time.sleep(1)
    hardware.leds_off()
    
    print("\n✅ Sistema listo para capturar")
    print("   Presione el botón para capturar una imagen")
    print("   Presione Ctrl+C para salir\n")
    
    try:
        while True:
            # Esperar botón
            hardware.wait_for_button()
            
            # LED azul: procesando
            hardware.set_led("BLUE")
            print("\n📷 Capturando imagen...")
            
            # Capturar imagen
            image = camera.capture()
            print(f"✅ Imagen capturada: {image.size}")
            
            # Enviar a servidor
            print("📤 Enviando al servidor para análisis...")
            result = client.analyze_image(
                image=image,
                location="Captura Raspberry Pi",
                notes="Captura automática desde dispositivo"
            )
            
            # Mostrar resultado
            print_result(result)
            
            # Controlar LED según resultado
            if result.get('success'):
                led_color = result.get('led_signal', 'BLUE')
                hardware.blink_led(led_color, times=3)
                hardware.set_led(led_color)
                time.sleep(3)
            else:
                hardware.blink_led("RED", times=5)
            
            hardware.leds_off()
            print("\n👆 Listo para nueva captura...")
            
    except KeyboardInterrupt:
        print("\n\n👋 Cerrando sistema...")
    finally:
        hardware.cleanup()
        camera.close()
        print("✅ Sistema cerrado correctamente")


if __name__ == "__main__":
    main()
