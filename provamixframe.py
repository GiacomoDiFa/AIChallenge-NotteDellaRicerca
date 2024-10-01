# prendere immagine vignetta
# renderla a massimo schermo in base allo schermo disponibile
# appiccicarci sopra il frame della webcam
# Python code to read image
import cv2
import pyautogui
import numpy as np
from PIL import Image

# To read image from disk, we use
# cv2.imread function, in below method,
cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
screen_width, screen_height = pyautogui.size()
#cap = cv2.VideoCapture(-1)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#ret, frame = cap.read()
#frame = cv2.flip(frame,1)

frame = Image.open("frame.png")


img = Image.open("bitmap.png")

img = img.resize(frame.size)

final = Image.composite(img,frame,img)

final.show()
cv2.imshow("frame",np.array(final))
#cv2.imshow("frame", img)

# To hold the window on screen, we use cv2.waitKey method
# Once it detected the close input, it will release the control
# To the next line
# First Parameter is for holding screen for specified milliseconds
# It should be positive integer. If 0 pass an parameter, then it will
# hold the screen until user close it.
cv2.waitKey(0)

# It is for removing/deleting created GUI window from screen
# and memory
cv2.destroyAllWindows()
