import os
import dlib
import cv2
from PIL import Image
import numpy as np

def get_img_data():
	paths = [os.path.join("Datasets/jones_gabriel",f) for f in os.listdir("Datasets/jones_gabriel")]
	faces = []
	ids = []
	for path in paths:
		image = Image.open(path).convert("L") #convert to grayscale
		image_np = np.array(image,"uint8")
		id = int(path.split(".")[1])
		
		ids.append(id)
		faces.append(image_np)
		
	return np.array(ids), faces
	
ids, faces = get_img_data()
lbph_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_classifier.train(faces,ids)
lbph_classifier.write("lbph_classifier.yml")

lbph_face_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_face_classifier.read("lbph_classifier.yml")

image = Image.open("Datasets/jones_gabriel/person.1.1.jpg")
print(image.size)

paths = [os.path.join("Datasets/jones_gabriel",f) for f in os.listdir("Datasets/jones_gabriel")]
for path in paths:
	image = Image.open(path).convert("L") #convert to grayscale
	image_np = np.array(image,"uint8")
	prediction, _ = lbph_face_classifier.predict(image_np)
	expected_output = int(path.split(".")[1])
	
	cv2.putText(image_np, "Pred: " + str(prediction), (10,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))
	cv2.putText(image_np, "Exp: " + str(expected_output), (10,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))
	cv2.imshow("",image_np)
	cv2.waitKey(0)