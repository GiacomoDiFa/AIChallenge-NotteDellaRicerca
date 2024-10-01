import cv2
from ultralytics import YOLO
import numpy as np
import threading
import tkinter as tk
import time
from PIL import Image, ImageTk
import pygame

# Inizializzazione della webcam e del modello YOLO
cap = cv2.VideoCapture(-1)
pygame.mixer.init()

model = YOLO("yolov8n.pt")
sound = pygame.mixer.Sound("bomb_siren.wav")

detection_number = 0
max_detection_value = 5

# Inizializzazione di Tkinter
root = tk.Tk()
root.title("Detection Display")

# Dimensione dello schermo
screen_size = (640, 480)

# Label per mostrare il frame e il colore dello schermo
frame_label = tk.Label(root)
frame_label.pack()

screen_label = tk.Label(root, width=300, height=300)
screen_label.pack()

# Lock per sincronizzare i thread
detection_lock = threading.Lock()

def capture_and_detect():
    global detection_number

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Errore: impossibile catturare il frame")
            break

        result = model(frame, classes=[0], conf=0.5)
        array = result[-1].boxes.xyxy.cpu()

        with detection_lock:
            if len(array) > 0:
                detection_number += 1
                sound.play()

        # Disegna i rettangoli attorno agli oggetti rilevati
        for i in array:
            frame = cv2.rectangle(frame, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (0, 255, 0), 2)

        # Converti l'immagine per Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        frame_label.imgtk = imgtk
        frame_label.configure(image=imgtk)

        # Aggiorna la GUI
        root.update_idletasks()

def manage_screen():
    global detection_number

    while True:
        with detection_lock:
            if detection_number >= max_detection_value:
                color = "red"  # Schermo rosso
                detection_number = 0
            else:
                color = "green"  # Schermo verde
        time.sleep(5)
        # Aggiorna il colore dello schermo
        screen_label.config(bg=color)

        # Aggiorna la GUI
        root.update_idletasks()

# Creazione dei thread
capture_thread = threading.Thread(target=capture_and_detect)
screen_thread = threading.Thread(target=manage_screen)

# Avvio dei thread
capture_thread.start()
screen_thread.start()

# Inizia il main loop di Tkinter
root.mainloop()

# Pulizia delle risorse
cap.release()