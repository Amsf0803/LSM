import cv2
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import math
import os
from gtts import gTTS
from datetime import datetime
import subprocess


# Configuración de MediaPipe para la detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Inicialización de la cámara 0 es la camara default, si tienes mas camaras puedes cambiarlo

cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Error: No se puede abrir la cámara.")
    exit()

# Variables para suavizado en el tiempo
consistency_count = 0
prev_detected_letter = None
last_spoken_letter = None
FRAMES_THRESHOLD = 5  # Número de cuadros consecutivos requeridos

# Índices de los landmarks
MCP = [5, 9, 13, 17]  # Índices de los MCP de los dedos
PIP = [6, 10, 14, 18]  # Índices de los PIP de los dedos
DIP = [7, 11, 15, 19]  # Índices de los DIP de los dedos

# Crear el directorio Audios si no existe
os.makedirs("Audios", exist_ok=True)


def calculate_angle(a, b, c):
    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)
    dot_prod = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    return math.degrees(math.acos(dot_prod / (mag_ba * mag_bc))) if mag_ba * mag_bc != 0 else 0

def is_finger_extended(landmarks, mcp_idx, pip_idx, dip_idx, threshold=160):
    return calculate_angle(landmarks[mcp_idx], landmarks[pip_idx], landmarks[dip_idx]) > threshold

def is_thumb_extended(landmarks, threshold=150):
    return calculate_angle(landmarks[2], landmarks[3], landmarks[4]) > threshold

def detect_vowel(hand_landmarks):
    landmarks = hand_landmarks.landmark
    finger_status = {
        "index": is_finger_extended(landmarks, MCP[0], PIP[0], DIP[0]),
        "middle": is_finger_extended(landmarks, MCP[1], PIP[1], DIP[1]),
        "ring": is_finger_extended(landmarks, MCP[2], PIP[2], DIP[2]),
        "pinky": is_finger_extended(landmarks, MCP[3], PIP[3], DIP[3]),
        "thumb": is_thumb_extended(landmarks)
    }

    # Mapeo de letras
    if (not finger_status["index"] and not finger_status["middle"] and 
        not finger_status["ring"] and not finger_status["pinky"] and finger_status["thumb"]):
        return "A"
    elif all(finger_status.values()):
        return "5"
    elif (finger_status["pinky"] and not any(finger_status[f] for f in ["index", "middle", "ring", "thumb"])):
        return "I"
    elif (finger_status["index"] and finger_status["middle"] and 
          not finger_status["ring"] and not finger_status["pinky"] and not finger_status["thumb"]):
        return "2"
    elif (finger_status["index"] and finger_status["middle"] and 
          not finger_status["ring"] and not finger_status["pinky"] and finger_status["thumb"]):
        return "3"
    elif (finger_status["index"] and finger_status["middle"] and 
          finger_status["ring"] and finger_status["pinky"] and not finger_status["thumb"]):
        return "4"
    elif (finger_status["index"] and not any(finger_status[f] for f in ["middle", "ring", "pinky", "thumb"])):
        return "1"
    elif (finger_status["index"] and finger_status["thumb"] and not any(finger_status[f] for f in ["middle", "ring", "pinky"])):
        return "L"
    elif (finger_status["index"] and finger_status["middle"] and 
        finger_status["ring"] and not finger_status["pinky"] and not finger_status["thumb"]):
        return "6"
    elif (finger_status["index"] and finger_status["middle"] and not
        finger_status["ring"] and  finger_status["pinky"] and not finger_status["thumb"]):
        return "7"
    elif (finger_status["index"] and not finger_status["middle"] and 
        finger_status["ring"] and  finger_status["pinky"] and not finger_status["thumb"]):
        return "8"
    elif (not finger_status["index"] and finger_status["middle"] and 
        finger_status["ring"] and  finger_status["pinky"] and not finger_status["thumb"]):
        return "9"
    elif (finger_status["pinky"] and finger_status["thumb"] and not any(finger_status[f] for f in ["index", "middle", "ring"])):
        return "Y"
    elif (finger_status["index"] and finger_status["pinky"] and finger_status["thumb"] and not any(finger_status[f] for f in ["middle", "ring"])):
        return "Te amo"
    return None

def detect_word(detectado):
    vocales = ["a", "e", "i", "o", "u", "A", "E", "I", "O", "U"]
    letra = "bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ"
    number = "0123456789"
    for vocal in vocales:
        if detectado == vocal:
            return "Vocal detectada:"
    for letras in letra:
        if detectado == letras:
            return "Letra detectada:"
    
    for num in number:
        if detectado == num:
            return "Numero detectado:"
    return "Palabra detectada:"

# Configuración de la ventana de Tkinter
window = tk.Tk()
window.title("Detección de Lenguaje de Señas Beta")
video_label = tk.Label(window)
video_label.pack()

def update_frame():
    global consistency_count, prev_detected_letter, last_spoken_letter, audio_counter
    ret, frame = cap.read()
    if not ret:
        window.after(10, update_frame)
        return

    frame = cv2.flip(frame, 1)
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    current_letter = None
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            letter = detect_vowel(handLms)
            word = detect_word(letter)
            if letter:
                current_letter = letter
                cv2.putText(frame, f'{word} {letter}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if current_letter == prev_detected_letter and current_letter is not None:
        consistency_count += 1
    else:
        consistency_count = 1
        prev_detected_letter = current_letter

    if consistency_count >= FRAMES_THRESHOLD and current_letter is not None and current_letter != last_spoken_letter:
        print(f"{word} {current_letter}")
        last_spoken_letter = current_letter
        
        # Obtener la fecha y hora actuales
        ahora = datetime.now()
        fecha_hora = ahora.strftime("%d-%m-%Y_%H:%M:%S")  # Formato: YYYY-MM-DD_HH:MM:SS
        
        # Crear un nombre de archivo único
        audio_file = f"Audios/audio_{fecha_hora}.mp3"


        tts = gTTS(text=current_letter, lang='es')
        tts.save(audio_file)
        # Lista de reproductores de audio
        reproductores = [
            [r"C:\Program Files (x86)\Windows Media Player\wmplayer.exe", audio_file],
            [r"C:\Program Files\Windows Media Player\wmplayer.exe", audio_file],
            ["mpg321", audio_file],
            ["afplay", audio_file]
        ]

        # Intentar reproducir el audio con cada reproductor
        for reproductor in reproductores:
            try:
                subprocess.run(reproductor, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"Reproduciendo con: {reproductor[0]}")
                break  # Salir del bucle si se reproduce correctamente
            except Exception as e:
                pass

    imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    
    window.after(10, update_frame)

# Iniciar el ciclo de actualización de la imagen
update_frame()
window.mainloop()

cap.release()



