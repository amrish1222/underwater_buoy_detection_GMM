# -*- coding: utf-8 -*-

import cv2

trainingImages = "CroppedImages/"

cap = cv2.VideoCapture('detectbuoy.avi')

i = 0
while(cap.isOpened()):
    _, frame = cap.read()
    
    cv2.imshow("img", frame)
    
    if cv2.waitKey(500) & 0xFF == ord('s'):
        i +=1
        path = trainingImages + "Img_" + str(i) + ".png"
        print(path)
        cv2.imwrite(path,frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  

