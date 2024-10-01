import cv2
import numpy as np

def mouse_callback(event,x,y,flags,param):
	if event == cv2.EVENT_LBUTTONDOWN:
		if button_x1 <= x <= button_x2 and button_y1 <= y <= button_y2:
			print("Bottone cliccato")

button_x1,button_y1 = 50,50
button_x2,button_y2=200,100

image = np.zeros((300,400,3),dtype=np.uint8)

cv2.rectangle(image,(button_x1,button_y1),(button_x2,button_y2),(0,255,0),-1)
cv2.putText(image,"Click me",(button_x1+20,button_y1+30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

cv2.namedWindow("Image")
cv2.setMouseCallback("Image",mouse_callback)

while True:
	cv2.imshow("Image",image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cv2.destroyAllWindows()