import cv2
import numpy as numpy
import os

def cut_faces(image, faces_coord):
	faces=[]

	for (x, y, w, h) in faces_coord:
		w_rm=int(0.2*w/2)
		faces.append(image[y:y+h, x+w_rm: x+w-w_rm  ])

	return faces
	
def normalize_intesity(images):
	images_norm=[]
	for image in images:
		is_color=len(image.shape)==3
		if is_color:
			image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		images_norm.append(cv2.equalizeHist(image))
	return images_norm


def resize(images, size=(50, 50)):
	images_norm=[]

	for image in images:
		if image.shape<size:
			image_norm=cv2.resize(image, size, interpolation=cv2.INTER_AREA)
		else:
			image_norm=cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
		images_norm.append(image_norm)
		
	return images_norm



def normalize_faces(frame, faces_coord):
	faces=cut_faces(frame, faces_coord)
	faces=normalize_intesity(faces)
	faces=resize(faces)
	return faces

def draw_rectangle(image, coords):
	for (x,y,w,h) in coords:
		w_rm=int(0.2*w/2)
		cv2.rectangle(image,( x+w_rm, y), (x+w-w_rm, y+h),(150, 150 , 0), 8)

cap=cv2.VideoCapture(0)
face=cv2.CascadeClassifier('face.xml')

folder="people/"+input('Person: ').lower() 
#cv2.namedWindow("Python Face Recognition", cv2.WINDOW_AUTOSIZE)

if not os.path.exists(folder):
	os.mkdir(folder)
	counter=0
	timer=0
	while counter<10:
		ret, frame=cap.read()
		gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces_coord=face.detectMultiScale(gray, 1.3, 5)
		if len(faces_coord) and timer%700==50:
 			faces=normalize_faces(frame, faces_coord)
 			cv2.imwrite(folder+"/"+str(counter)+".jpg",faces[0])
 			cv2.imshow(str(counter),faces[0])
 			counter+=1
		draw_rectangle(frame, faces_coord)
		cv2.imshow('Face it',frame)
		cv2.waitKey(50)
		timer+=50
	cv2.destroyAllWindows()
else:
	print("This name already exists")    

cap.release()	