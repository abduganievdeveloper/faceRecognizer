import cv2
import numpy as numpy

cap=cv2.VideoCapture(0)
face_haar=cv2.CascadeClassifier('face.xml')


while True:
  ret, frame=cap.read()
  gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces=face_haar.detectMultiScale(gray, 1.3 ,5)
  
  for (x,y,w,h) in faces:
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
  
  cv2.imshow('frame', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break