import os
import dlib
import cv2
from PIL import Image
import numpy as np

face_detector = dlib.get_frontal_face_detector()
points_detector = dlib.shape_predictor("Weights/shape_predictor_68_face_landmarks.dat")
face_descriptor_extractor = dlib.face_recognition_model_v1("Weights/dlib_face_recognition_resnet_model_v1.dat")

index = {}
idx = 0
face_descriptors = None

paths = [os.path.join("Datasets/yalefaces/train",f) for f in os.listdir("Datasets/yalefaces/train")]
for path in paths:
	#print(path)
	image = Image.open(path).convert("RGB") #with dlib, no need to convert to grayscale
	image_np = np.array(image,"uint8")
	face_detection = face_detector(image_np,1)
	for face in face_detection:
		l,t,r,b = face.left(),face.top(),face.right(),face.bottom()
		cv2.rectangle(image_np,(l,t),(r,b),(0,0,255),2)
		
		points = points_detector(image_np,face)
		for point in points.parts():
			cv2.circle(image_np, (point.x, point.y), 2, (0,255,0), 1)
	cv2.imshow("",image_np)
	
	cv2.waitKey(0)

	