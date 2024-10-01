# YOLO-based Object Detection Game

Questo progetto implementa un semplice gioco di rilevamento degli oggetti basato su YOLO (You Only Look Once) con OpenCV e Pygame. L'applicazione rileva persone utilizzando il modello YOLOv8 e proietta il gioco su uno schermo. Quando il numero di rilevazioni supera una soglia predefinita, il gioco termina con un suono "game over".

## Prerequisiti

Prima di iniziare, assicurati di avere installato i seguenti pacchetti:

- Python 3.x
- OpenCV
- PyGame
- PyAutoGUI
- Numpy
- Ultralytics YOLO

Puoi installare le dipendenze eseguendo:

```bash
pip install opencv-python pygame pyautogui numpy ultralytics
```
Si consiglia di utilizzare un ambiente virtuale

## Descrizione del progetto

Il gioco funziona tramite due thread:

1. **Main Thread**: Questo thread acquisisce i frame dalla webcam, esegue la rilevazione degli oggetti utilizzando YOLO e visualizza le immagini con le bounding box proiettate a schermo. Controlla anche se il numero di rilevamenti supera una soglia predefinita, e in tal caso, segnala la perdita e riproduce un suono di "game over".

2. **Detection Counter & IOU Thread**: Questo thread calcola l'Intersection Over Union (IOU) tra la bounding box rilevata nel frame corrente e quella del frame precedente. Se l'IOU è inferiore a una soglia, incrementa il contatore delle rilevazioni dell'applicazione e riproduce un segnale acustico.

## Suoni

Il gioco utilizza due suoni:
- `beep.wav`: Riprodotto ogni volta che viene fatta una nuova rilevazione.
- `game_over_sound_effect.wav`: Riprodotto quando il numero di rilevazioni supera la soglia massima.

## Costanti configurabili

Puoi modificare due parametri principali nel codice:
- **IOU_THRESHOLD**: Soglia per il calcolo dell'IOU (default: 0.85).

## Come eseguire il codice

1. Attiva l'ambiente virtuale con il comando `source venv/bin/activate` e assicurati di avere tutti i pacchetti necessari
2. Assicurati di avere i suoni `beep.wav` e `game_over_sound_effect.wav` nella directory del progetto.
3. Esegui lo script principale:

```bash
python game.py
```

Il gioco partirà, utilizzando la tua webcam per rilevare le persone nel frame.

## Tasti di controllo

- **`q`**: Esci dal gioco.
- **`r`**: Resetta le variabili per ricominciare una nuova partita.

## Struttura del codice

- **`SharedData`**: Classe che gestisce le variabili condivise tra i thread.
- **`reset_variables`**: Funzione che resetta le variabili di gioco.
- **`main_thread`**: Thread principale che gestisce la cattura dei frame e la logica di gioco.
- **`detection_counter_iou_thread`**: Thread che calcola l'IOU e incrementa il contatore delle rilevazioni.
- **`calculate_IOU`**: Funzione che calcola l'IOU tra due bounding box.

## Setup per il gioco
Posizionare la Webcam a **due metri dalla parete** ad un'**altezza di un metro e quaranta**.
Il **percorso del gioco** deve essere di **due metri e quaranta di lunghezza**.

## Oggetti per evitare di essere rilevati da YOLO
- **Coperte/mantelli** 
- **Ombrelli grandi**
- **Tessuti con pattern confusi**
- **Cartoni o scatole (magari anche dipinti)**
- **Specchi o superfici riflettenti** 
- **Zaini o borse grandi** 
- **Cuscini**

## Conclusione

Questo progetto è un semplice esempio di come utilizzare YOLO per creare un gioco interattivo con Python, usando tecniche di threading e variabili condivise.
Sentiti pure libero di cambiare il codice come meglio credi :)



game2 è il gioco ufficiale
game3 è il gioco con i suoni e il reset automatico (buggato se si preme r ma utile per digit)
game4 è il gioco con il fish eye