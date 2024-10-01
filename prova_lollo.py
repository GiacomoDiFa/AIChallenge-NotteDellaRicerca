import cv2
from ultralytics import YOLO
import numpy as np
import pygame
import threading
import time
import pyautogui

def main_thread():
    global iou_lock, iou, detection_number, application_detections, soluzione,old_detection
    
    def reset_variables():
        global detection_number, application_detections
        with detection_lock:
            detection_number = 0
            application_detections = 0
        sound2.stop()
        print("Variabili resettate")
    
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


    # Inizializzazione della webcam
    cap = cv2.VideoCapture(-1)
    # Dimensione dello schermo
    screen_width, screen_height = pyautogui.size()
    # Inizializzo il modello
    model = YOLO("yolov8n.pt")

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

        result = model(frame,classes=[0],conf=0.5)
        array = result[-1].boxes.xyxy.cpu()

        if len(array) > 0:
            with detection_lock:
                print('soluzione:',len(array))
                detection_number += 1
                print(detection_number)
            current_detection = (int(array[0][0]),int(array[0][1]),int(array[0][2]),int(array[0][3]))
            if old_detection is None:
                old_detection = (int(array[0][0]),int(array[0][1]),int(array[0][2]),int(array[0][3]))
            tmp_iou = calculate_IOU(old_detection, current_detection)
            old_detection = current_detection
            with iou_lock:
                iou = 1
            frame = cv2.rectangle(frame, (int(array[0][0]),int(array[0][1])),(int(array[0][2]),int(array[0][3])), (0, 255, 0), 2)
        with detection_lock:
            if application_detections >= max_detection_value:
                print('perso')
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), -1)
                frame = cv2.addWeighted(overlay, 0.5, frame, 1 - 0.5, 0)
                sound1.stop()
                sound2.play()


        resized_image = cv2.resize(frame,(screen_width,screen_height))
        cv2.imshow("frame", resized_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            reset_variables()


    # Esci dal thread
    cap.release()
    cv2.destroyAllWindows()

def detection_counter_thread():
    global iou_lock, iou, soglia_IOU,detection_number,application_detections
    i=0
    while True:
        with iou_lock:
            #print(f"{iou=} {soglia_IOU=}")
            dormire = False if iou < soglia_IOU else True
        if dormire:
            time.sleep(1)
            print('sto dormendo')
        #print(f"{i} CIAO")
        i+=1
        with detection_lock:
            if detection_number>=1:
                application_detections+=1
                print('application detections:',application_detections)
            detection_number=0



# Caricamento dei suoni
pygame.mixer.init()
sound1 = pygame.mixer.Sound("beep.wav")
sound2 = pygame.mixer.Sound("game_over_sound_effect.wav")

# variabili condivise
detection_number = 0
application_detections = 0
detection_lock = threading.Lock()
iou_lock = threading.Lock()
old_detection = None
iou = 0
# Costanti
max_detection_value = 10
soglia_IOU =0.1
#soluzione = 2

main_thread = threading.Thread(target=main_thread)
main_thread.start()
dc_thread = threading.Thread(target=detection_counter_thread)
dc_thread.start()

main_thread.join()
dc_thread.join()

