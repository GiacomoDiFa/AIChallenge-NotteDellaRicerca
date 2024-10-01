import cv2
from ultralytics import YOLO
import numpy as np
import pygame
import threading
import pyautogui

# Variabili condivise
class SharedData:
    def __init__(self):
        self.application_detections = 0
        self.detection_list = []
        self.detection_lock = threading.Lock()
        self.detection_semaphore = threading.Semaphore(0)
        self.perso = 0

# Caricamento dei suoni
pygame.mixer.init()
sound1 = pygame.mixer.Sound('beep.wav')
sound2 = pygame.mixer.Sound('game_over_sound_effect.wav')

# Costanti
MAX_DETECTION_VALUE = 10
IOU_THRESHOLD = 0.85

# Costanti testate con 2 metri dalla parete, 2.40 metri di lunghezza della pista, 1.40 metri di altezza
#IOU_THRESHOLD = 0.85
#MAX_DETECTION_VALUE = 10

def reset_variables(shared_data):
    with shared_data.detection_lock:
        shared_data.application_detections = 0
        shared_data.perso = 0
    print('Variabili resettate')
    

def main_thread(shared_data):
    # Inizializzazione della webcam
    cap = cv2.VideoCapture(-1)
    
    # Dimensione dello schermo
    screen_width, screen_height = pyautogui.size()
    
    # Inizializzo il modello
    model = YOLO("yolov8n.pt")

    if not cap.isOpened():
        print("Errore: impossibile aprire la webcam")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Errore: impossibile catturare il frame")
            break

        result = model(frame, classes=[0], conf=0.5)
        array = result[-1].boxes.xyxy.cpu()

        if len(array) > 0:
            #acquisisco il lock
            # metto dentro shared data la detection
            #fai la release sul semaforo contatore
            #rilascio il lock
            with shared_data.detection_lock:
                if len(shared_data.detection_list) < 2:
                    shared_data.detection_list.append((int(array[0][0]), int(array[0][1]), int(array[0][2]), int(array[0][3])))
                    shared_data.detection_list.append((int(array[0][0]), int(array[0][1]), int(array[0][2]), int(array[0][3])))
                else:
                    shared_data.detection_list.append((int(array[0][0]), int(array[0][1]), int(array[0][2]), int(array[0][3])))
                    shared_data.detection_list.pop(0)
                shared_data.detection_semaphore.release()
            frame = cv2.rectangle(frame, (int(array[0][0]), int(array[0][1])), (int(array[0][2]), int(array[0][3])), (0, 255, 0), 2)
        
        with shared_data.detection_lock:
            #qua serve solo lock
            if shared_data.application_detections >= MAX_DETECTION_VALUE:
                print('perso')
                shared_data.perso += 1
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), -1)
                frame = cv2.addWeighted(overlay, 0.5, frame, 1 - 0.5, 0)
                if shared_data.perso == 1:
                    sound1.stop()
                    sound2.play()

        resized_image = cv2.resize(frame, (screen_width, screen_height))
        cv2.imshow("frame", resized_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            reset_variables(shared_data)

    cap.release()
    cv2.destroyAllWindows()

def detection_counter_iou_thread(shared_data):
    def calculate_IOU(box1,box2):
        #x1,y1 è top left di box1
        #x2,y2 è bottom right di box1
        #x3,y3 è top left di box2
        #x4,y4 è bottom right di box2

        x1,y1,x2,y2 = box1
        x3,y3,x4,y4 = box2
        x_inter1 = max(x1,x3)
        y_inter1 = max(y1,y3)
        x_inter2 = min(x2,x4)
        y_inter2 = min(y2,y4)
        width_inter = abs(x_inter2 - x_inter1)
        height_inter = abs(y_inter2 - y_inter1)
        area_inter = width_inter * height_inter
        width_box1 = abs(x2-x1)
        height_box1 = abs(y2-y1)
        width_box2 = abs(x4-x3)
        height_box2 = abs(y4-y3)
        area_box1 = width_box1 * height_box1
        area_box2 = width_box2 * height_box2
        area_union = area_box1 + area_box2 - area_inter
        iou = area_inter / area_union
        return iou

    while True:
        shared_data.detection_semaphore.acquire()
        with shared_data.detection_lock:
            if len(shared_data.detection_list) >= 2:
                #acquisire semafoto contatore
                #acquisice il lock per vedere lista
                #calcola iou e fa tutte le altre cose
                #rilascia il lock
                iou = calculate_IOU(shared_data.detection_list[1], shared_data.detection_list[0])
                print("iou:",iou)
                if iou < IOU_THRESHOLD:
                    shared_data.application_detections += 1
                    sound1.play()
                
            

# Crea l'oggetto condiviso
shared_data = SharedData()

# Creazione dei thread
main_thread_instance = threading.Thread(target=main_thread, args=(shared_data,))
dc_thread_instance = threading.Thread(target=detection_counter_iou_thread, args=(shared_data,))

# Avvia i thread
main_thread_instance.start()
dc_thread_instance.start()

# Attendi che i thread terminino
main_thread_instance.join()
dc_thread_instance.join()