import cv2
from ultralytics import YOLO
import numpy as np
import pygame
import threading
import time

# Inizializzazione del modello e della webcam
cap = cv2.VideoCapture(-1)
pygame.mixer.init()

# Caricamento dei suoni
sound1 = pygame.mixer.Sound("beep.wav")
sound2 = pygame.mixer.Sound("bomb_siren.wav")

model = YOLO("yolov8n.pt")

# Variabili globali per la gestione delle detections
detection_number = 0
application_detections = 0
max_detection_value = 10
frame_skipper = 0
max_frame_skipper_value = 4

detection_lock = threading.Lock()

# Funzione che conta le detections della rete
def detection_tracker():
    global detection_number, application_detections

    while True:
        time.sleep(1)
        with detection_lock:
            if detection_number > 1:
                application_detections += 1
                sound1.play()
            detection_number = 0  # Resetta il contatore ogni 10 secondi

# Thread per la gestione delle detections
tracker_thread = threading.Thread(target=detection_tracker)
tracker_thread.start()

def reset_variables():
    global detection_number, application_detections
    with detection_lock:
        detection_number = 0
        application_detections = 0
    sound2.stop()
    print("Variabili resettate")

if not cap.isOpened():
    print("Errore: impossibile aprire la webcam")
else:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()

    if not ret:
        print("Errore: impossibile catturare il frame")
        break

    
    
    if frame_skipper < max_frame_skipper_value:
        frame_skipper += 1
        with detection_lock:
            print(application_detections)
            if application_detections >= max_detection_value:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), -1)
                frame = cv2.addWeighted(overlay, 0.5, frame, 1 - 0.5, 0)
                sound1.stop()
                sound2.play()   

    else:
        result = model(frame, classes=[0], conf=0.5)
        array = result[-1].boxes.xyxy.cpu()

        if len(array) > 0:
            with detection_lock:
                detection_number += 1

        for i in array:
            frame = cv2.rectangle(frame, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (0, 255, 0), 2)

        with detection_lock:
            print(application_detections)
            if application_detections >= max_detection_value:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), -1)
                frame = cv2.addWeighted(overlay, 0.5, frame, 1 - 0.5, 0)
                sound1.stop()
                sound2.play()
        frame_skipper = 0

    cv2.imshow("frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        reset_variables()
    
cap.release()
cv2.destroyAllWindows()
tracker_thread.join()

