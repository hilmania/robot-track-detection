import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("videos/dataset-1.mp4")

cap.set(3, 640)
cap.set(4, 480)

segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()

while True:
    success, img = cap.read()
    imgOut = segmentor.removeBG(img, (255,0,255), threshold=0.83)
    imgStack = cvzone.stackImages([img, imgOut], 2, 1)
    _, imgStack = fpsReader.update(imgStack)
    cv2.imshow("Image", imgStack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
