import cv2
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import math
import os
from gtts import gTTS
import threading
from playsound import playsound
import time

# --- CONFIGURACIÓN ---
mp_hands = mp.solutions.hands
# Aumentamos confianza para evitar falsos positivos
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Crear carpeta de audios
if not os.path.exists("Audios"):
    os.makedirs("Audios")

# Variables globales de estado
state = {
    "consistency_count": 0,
    "prev_char": None,
    "last_spoken": None,
    "text_buffer": "" # Para ir formando palabras (opcional)
}

FRAMES_THRESHOLD = 10  # Cuántos frames debe mantenerse la seña para validarla

# --- MOTOR DE AUDIO OPTIMIZADO ---
def play_audio_threaded(text):
    """
    Sistema inteligente de audio:
    1. Verifica si el archivo ya existe (Cache).
    2. Si no existe, lo descarga de Google (gTTS).
    3. Lo reproduce en un hilo para no trabar el video.
    """
    file_path = f"Audios/{text}.mp3"

    def _worker():
        try:
            # Solo generar si no existe
            if not os.path.exists(file_path):
                print(f"Generando audio nuevo para: {text}...")
                tts = gTTS(text=text, lang='es')
                tts.save(file_path)
            
            # Reproducir
            playsound(file_path)
        except Exception as e:
            print(f"Error de audio: {e}")

    # Ejecutar en hilo separado
    threading.Thread(target=_worker, daemon=True).start()

# --- LÓGICA DE GEOMETRÍA ---
def calculate_angle(a, b, c):
    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)
    dot_prod = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    if mag_ba * mag_bc == 0:
        return 0
    angle = math.degrees(math.acos(min(1.0, max(-1.0, dot_prod / (mag_ba * mag_bc)))))
    return angle

def get_finger_status(lm):
    # Índices de las articulaciones (MCP, PIP, DIP, TIP)
    # Pulgar es especial, comprobamos ángulo y posición x respecto al índice
    fingers = {}
    
    # --- PULGAR (Cálculo más complejo para precisión) ---
    # Ángulo entre muñeca, MCP y punta
    thumb_angle = calculate_angle(lm[0], lm[2], lm[4])
    # También chequeamos si la punta del pulgar está lejos de la palma (eje x/y dependiendo de mano)
    fingers["thumb"] = thumb_angle > 150 # Simplificado para este ejemplo

    # --- OTROS DEDOS (Ángulo de articulación PIP) ---
    # Índice
    angle_index = calculate_angle(lm[5], lm[6], lm[8])
    fingers["index"] = angle_index > 160

    # Medio
    angle_middle = calculate_angle(lm[9], lm[10], lm[12])
    fingers["middle"] = angle_middle > 160

    # Anular
    angle_ring = calculate_angle(lm[13], lm[14], lm[16])
    fingers["ring"] = angle_ring > 160

    # Meñique
    angle_pinky = calculate_angle(lm[17], lm[18], lm[20])
    fingers["pinky"] = angle_pinky > 160

    return fingers

def interpret_sign(fingers, lm):
    """
    Diccionario de reglas. 
    True = Dedo estirado
    False = Dedo doblado
    """
    f = fingers # Alias corto
    
    # Patrones (Thumb, Index, Middle, Ring, Pinky)
    # Nota: El pulgar es el más difícil de detectar consistentemente con solo ángulos,
    # a veces requiere verificar cercanía con otros dedos.
    
    if f["index"] and f["middle"] and f["ring"] and f["pinky"] and f["thumb"]:
        return "Hola / 5"
    
    if not f["index"] and not f["middle"] and not f["ring"] and not f["pinky"] and f["thumb"]:
        return "A" 
    if  f["index"] and  f["middle"] and not f["ring"] and not f["pinky"] and f["thumb"]:
        return "3"
    if f["index"] and not f["middle"] and not f["ring"] and not f["pinky"]:
        # Si el pulgar está abierto es L, si no es 1/D
        if f["thumb"]: return "L"
        else: return "1"

    if f["index"] and f["middle"] and not f["ring"] and not f["pinky"]:
        return "2" 
    
    if f["index"] and f["middle"] and f["ring"] and not f["pinky"]:
        return "6" 
    
    if f["pinky"] and f["thumb"] and not f["index"] and not f["middle"] and not f["ring"]:
        return "Y" # O gesto de llamar
    
    
    if f["thumb"] and f["index"] and f["pinky"] and not f["middle"] and not f["ring"]:
        return "Te amo"
    
    if f["middle"] and not f["index"] and not f["ring"] and not f["pinky"]:
        return "Grosero" # Dedo medio
    
    if not f["index"] and not f["middle"] and not f["ring"] and f["pinky"]:
        return "I"
    if  f["index"] and  f["middle"] and  f["ring"] and f["pinky"] and not f["thumb"]:
        return "4"
    if  f["index"] and  f["middle"] and not f["ring"] and f["pinky"] and not f["thumb"]:
        return "7"
    if  f["index"] and not f["middle"] and f["ring"] and f["pinky"] and not f["thumb"]:
        return "8"
    if not f["index"] and  f["middle"] and f["ring"] and f["pinky"] and not f["thumb"]:
        return "9"
    if not f["index"] and not  f["middle"] and not f["ring"] and not f["pinky"] and not f["thumb"]:
        return "E"
    return None

# --- INTERFAZ GRÁFICA ---
window = tk.Tk()
window.title("Traductor de Señas IA - Optimizado")
window.geometry("800x600")

# Estilos
label_title = tk.Label(window, text="Detectando...", font=("Arial", 24, "bold"))
label_title.pack(pady=10)

video_frame = tk.Label(window)
video_frame.pack()

label_status = tk.Label(window, text="Listo", font=("Arial", 14), fg="gray")
label_status.pack(pady=10)

# Inicializar cámara
cap = cv2.VideoCapture(0)

def update_loop():
    ret, frame = cap.read()
    if not ret:
        window.after(10, update_loop)
        return

    # Espejo y Color
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)
    
    current_sign = None
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 1. Obtener estado de los dedos
            fingers_status = get_finger_status(hand_landmarks.landmark)
            
            # 2. Interpretar seña
            current_sign = interpret_sign(fingers_status, hand_landmarks.landmark)
            
            # Mostrar texto en pantalla sobre la mano
            if current_sign:
                h, w, c = frame.shape
                cx, cy = int(hand_landmarks.landmark[9].x * w), int(hand_landmarks.landmark[9].y * h)
                cv2.putText(frame, current_sign, (cx - 50, cy - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # --- LÓGICA DE ESTABILIZACIÓN Y AUDIO ---
    if current_sign:
        if current_sign == state["prev_char"]:
            state["consistency_count"] += 1
        else:
            state["consistency_count"] = 0
            state["prev_char"] = current_sign
        
        # Umbral alcanzado y es una seña diferente a la última hablada
        if state["consistency_count"] > FRAMES_THRESHOLD:
            if current_sign != state["last_spoken"]:
                label_title.config(text=f"Seña: {current_sign}", fg="green")
                play_audio_threaded(current_sign)
                state["last_spoken"] = current_sign
                state["consistency_count"] = 0 # Resetear contador para no repetir inmediatamente
    else:
        # Si no hay mano o no hay seña reconocida, reseteamos el contador
        state["consistency_count"] = 0
        if not results.multi_hand_landmarks:
            # Opcional: Permitir repetir la misma palabra si sacas la mano y la vuelves a meter
            state["last_spoken"] = None 

    # Renderizar en Tkinter
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_frame.imgtk = imgtk
    video_frame.configure(image=imgtk)
    
    window.after(10, update_loop)

def on_closing():
    cap.release()
    window.destroy()

window.protocol("WM_DELETE_WINDOW", on_closing)
update_loop()
window.mainloop()