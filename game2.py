import cv2
from ultralytics import YOLO
import numpy as np
import pygame
import threading
import pyautogui
import json
import random
from PIL import ImageFont, ImageDraw, Image

# Variabili condivise
class SharedData:
    # Classe che implementa tutte le variabili condivise
    def __init__(self):
       
        # Detection dell'applicazione massime(!= da detection della rete YOLO)
        self.application_detections = 10
        
        # Lista delle detection eseguite dalla rete YOLO per il calcolo dell'IOU (dimensione massima 2)
        self.detection_list = []
        
        # Lock per le detection
        self.detection_lock = threading.Lock()
        
        # Semaforo per le detection per implementare pattern produttore-consumatore settato a 0 permessi
        self.detection_semaphore = threading.Semaphore(0)
        
        # Variabile per indicare se hai perso e far smettere di riprodurre il suono finale
        self.perso = 0

        # Variabile per il conteggio del numero di detection eseguita dalla rete YOLO
        self.detection_number = 0

        # Random text
        self.randomText = ""

# Caricamento dei suoni
pygame.mixer.init()
sound1 = pygame.mixer.Sound('second_beep.wav')
sound2 = pygame.mixer.Sound('game_over_sound_effect.wav')

# Caricamento dei font
# Primo font
font_path = "NotoSans-Medium.ttf"
font_size = 32
font = ImageFont.truetype(font_path,font_size)
# Secondo font
font_path_2 = "Oswald-Bold.ttf"
font_size_2 = 100
font2 = ImageFont.truetype(font_path_2,font_size_2)


# Caricamento dei testi
with open('texts.json','r') as file:
    texts = json.load(file)

# Costanti
#MAX_DETECTION_VALUE = 10 non più utilizzata
IOU_THRESHOLD = 0.9
MAX_DETECTION_NUMBER = 200

# Costanti testate con 2 metri dalla parete, 2.40 metri di lunghezza della pista, 1.40 metri di altezza
#IOU_THRESHOLD = 0.85
#MAX_DETECTION_VALUE = 10

def reset_variables(shared_data):
    '''
    Funzione per resettare le variabili application detections e perso.

    Parametri:
    - shared_data (SharedData): Istanza condivisa della classe SharedData.
    '''
    
    # Acquisisco il lock  
    with shared_data.detection_lock:
        shared_data.application_detections = 10
        shared_data.perso = 0
        shared_data.detection_number = 0
    # Rilascio il lock
    print('Variabili resettate')

    # Scelgo un testo in maniera casuale
    shared_data.randomText = random.choice(list(texts['texts']))
    shared_data.randomText = shared_data.randomText['text']
    

def main_thread(shared_data):
    '''
    Thread per l'acquisizione dei frame dalla webcam, la detection della rete YOLO e la visualizzazione a schermo del gioco.

    Parametri:
    - shared_data (SharedData): Istanza condivisa della classe SharedData
    '''

    # Inizializzazione della webcam
    cap = cv2.VideoCapture(-1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(width,height)

    
    # Dimensione dello schermo
    screen_width, screen_height = pyautogui.size()
    
    # Inizializzo il modello
    model = YOLO("yolov8n.pt")

    if not cap.isOpened():
        print("Errore: impossibile aprire la webcam")
        return
    cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Dimensione del frame acquisito dalla webcam
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Scelgo un testo in maniera casuale
    shared_data.randomText = random.choice(list(texts['texts']))
    shared_data.randomText = shared_data.randomText['text']

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)

        if not ret:
            print("Errore: impossibile catturare il frame")
            break
        
        # Inferenza della rete YOLO nel frame attuale filtrando solo la classe 0 (persone) con confindenza 0.5
        result = model(frame, classes=[0], conf=0.5)

        # Bounding box della detection
        array = result[-1].boxes.xyxy.cpu()

        # Controllo che ci siano detection nel frame attuale
        if len(array) > 0:
            # Incremento il numero delle detection eseguite dalla rete
            shared_data.detection_number += 1
            # Acquisisco il lock
            with shared_data.detection_lock:
                # Siccome voglio calcolare l'IOU tra il frame precedente e il frame attuale controllo se la lunghezza della lista e' minore di 2
                if len(shared_data.detection_list) < 2:
                    # Appendo nella lista lo stesso frame due volte
                    shared_data.detection_list.append((int(array[0][0]), int(array[0][1]), int(array[0][2]), int(array[0][3])))
                    shared_data.detection_list.append((int(array[0][0]), int(array[0][1]), int(array[0][2]), int(array[0][3])))
                else:
                    # Appendo il frame attuale e rimuovo il primo elemento della lista per mantenere la lunghezza della lista uguale a 2
                    shared_data.detection_list.append((int(array[0][0]), int(array[0][1]), int(array[0][2]), int(array[0][3])))
                    shared_data.detection_list.pop(0)
                # Rilascio un permesso per il semaforo
                shared_data.detection_semaphore.release()
            # Rilascio il lock

            # Disegno nel frame la bounding box dell' oggetto di cui e' stata fatta la detection
            frame = cv2.rectangle(frame, (int(array[0][0]), int(array[0][1])), (int(array[0][2]), int(array[0][3])), (0, 255, 0), 2)
        
        with shared_data.detection_lock:
            text_detection_application = f"Vite: {shared_data.application_detections}" 
        text_detection_model = f"Rilevamenti:{shared_data.detection_number}"
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #font_scale = 0.7
        #font_thickness = 2
        #color = (139,0,0)
        #text_size,_ = cv2.getTextSize(text_detection_model,font,font_scale,font_thickness)
        text_x = width - 147 - 10
        #print("text_sixe",text_size[0])
        text_y = 50

        frame_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame_pil)
        _,_,w1,h1 = draw.textbbox((0,0),text_detection_model,font=font)
        draw.rectangle(((10,25),(10+w1,25+h1)),fill="white")
        draw.text((10,25),text_detection_model,font=font,fill=(0,255,0,0))
        frame = np.array(frame_pil)
        
        frame_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame_pil)
        _,_,w2,h2 = draw.textbbox((0,0),text_detection_application,font=font)
        draw.rectangle(((text_x,25),(text_x+w2,25+h2)),fill="white")
        draw.text((text_x,25),text_detection_application,font=font,fill=(0,255,0,0))
        frame = np.array(frame_pil)
        


        # Acquisisco il lock
        with shared_data.detection_lock:
            # Controllo se le detection dell'applicazione siano uguali a zero (ho finito le vite)
            if shared_data.application_detections == 0 or shared_data.detection_number > MAX_DETECTION_NUMBER:
                print('Perso')
                # Incremento la variabile condivisa perso
                shared_data.perso += 1
                # Faccio diventare lo schermo rosso
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), -1)
                frame = cv2.addWeighted(overlay, 0.5, frame, 1 - 0.5, 0)
                #cv2.putText(frame,shared_data.randomText,(int(screen_width/2),int(screen_height/2)),font,font_scale,color,font_thickness)
                frame_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(frame_pil)
                _,_,w3,h3 = draw.textbbox((0,0),shared_data.randomText,font=font2)
                draw.text(((width-w3)/2,(height-h3)/2),shared_data.randomText,font=font2,fill=(255,255,255,0))
                frame = np.array(frame_pil)
                # Controllo se il gioco e' stato perso
                if shared_data.perso == 1:
                    # Fermo il primo suono (beep)
                    sound1.stop()
                    # Eseguo il secondo suono (game over)
                    sound2.play()
        # Rilascio il lock

        # Faccio il resize dell' immagine in modo che occupi tutto lo schermo nel quale l'applicazione viene eseguita e proiettata
        resized_image = cv2.resize(frame, (screen_width, screen_height))
        # Mostro a schermo l'immagine con il resize
        cv2.imshow("frame", resized_image)


        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Se premo il tasto 'r' da tastiera chiamo la funzione che resetta le variabili per ricominciare un nuovo gioco
            reset_variables(shared_data)

    cap.release()
    cv2.destroyAllWindows()

def detection_counter_iou_thread(shared_data):
    '''
    Thread per il calcolo dell'iou tra il frame precedente e il frame attuale e per l'incremento della variabile application detection.

    Parametri:
    - shared_data (SharedData): Istanza condivisa della classe SharedData.
    '''
    def calculate_IOU(box1,box2):
        '''
        Funzione per il calcolo dell'IOU tra due bounding box

        Parametri:
        - box1 (tupla): Tupla contenente (x1,y1,x2,y2) dove (x1,y1 e' il top left di box1) e (x2,y2 e' il bottom right di box1)
        - box2 (tupla): Tupla contenente (x3,y3,x4,y4) dove (x3,y3 e' il top left di box2) e (x4,y4 e' il bottom right di box2)
        '''
        
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
        # Acquisisco semaforo contatore
        shared_data.detection_semaphore.acquire()
        # Acquisisco il lock
        with shared_data.detection_lock:
            if len(shared_data.detection_list) >= 2:
                # Calcolo IOU
                iou = calculate_IOU(shared_data.detection_list[1], shared_data.detection_list[0])
                print("iou:",iou)
                # Controllo se IOU e' minore della soglia di gioco
                if iou < IOU_THRESHOLD:
                    # Controllo se il numero delle detection delle applicazioni è a zero (ho finito le vite)
                    if shared_data.application_detections == 0:
                        pass
                    else:
                        # Decremento la variabile condivisa application detections (ho perso una vita)
                        shared_data.application_detections -= 1
                        # Eseguo il primo suono (beep)
                        sound1.play()
        # Rilascio il lock
            

# Creazione dell'oggetto condiviso
shared_data = SharedData()

# Creazione dei thread
main_thread_instance = threading.Thread(target=main_thread, args=(shared_data,))
dc_thread_instance = threading.Thread(target=detection_counter_iou_thread, args=(shared_data,))

# Esecuzione dei thread
main_thread_instance.start()
dc_thread_instance.start()

# Attesa della terminazione dei thread
main_thread_instance.join()
dc_thread_instance.join()