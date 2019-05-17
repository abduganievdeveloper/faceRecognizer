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
    cv2.rectangle(image,( x+w_rm, y), (x+w-w_rm, y+h),(0, 150 , 0), 4)



def collect_dataset():
	images=[]
	labels=[]
	labels_dic={}
	people=[person for person in os.listdir("people/")]
	for i, person in enumerate(people):
		labels_dic[i]=person
		for image in os.listdir("people/"+person):
  

  			images.append(cv2.imread("people/"+person+'/'+image, 0))
  			labels.append(i)

	return (images, numpy.array(labels), labels_dic)




images, labels, labels_dic=collect_dataset()

rec_lbph=cv2.face.LBPHFaceRecognizer_create()
rec_lbph.train(images, labels)
print("Models trained successfully")

cap=cv2.VideoCapture(0)
facexml=cv2.CascadeClassifier('face.xml')

while True:
  ret, frame=cap.read()
  gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces_coord=facexml.detectMultiScale(gray, 1.3 ,5)


  if(len(faces_coord)):
    faces=normalize_faces(frame, faces_coord)
    for i, face in enumerate(faces):
    	collector=cv2.face.StandardCollector_create()
    	rec_lbph.predict_collect(face, collector)
    	conf=collector.getMinDist()
    	pred=collector.getMinLabel()
    	threshold=100
    	if conf<threshold:
    		cv2.putText(frame, 'Unknown'+str(conf), (faces_coord[i][0], faces_coord[i][1]-10), cv2.FONT_HERSHEY_PLAIN, 3,(66,53,243), 2 )
    	else:
    		cv2.putText(frame, labels_dic[pred].capitalize(), (faces_coord[i][0], faces_coord[i][1]-10), cv2.FONT_HERSHEY_PLAIN, 3,(66,53,243), 2 )
    draw_rectangle(frame, faces_coord)
  cv2.imshow('frame',frame)
 	
 
  if cv2.waitKey(1) & 0xFF == ord('q'):

    break

cap.release()		
cv2.destroyAllWindows()
