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
		face_descriptor = face_descriptor_extractor.compute_face_descriptor(image_np,points)
		#print(type(face_descriptor))
		#print(len(face_descriptor))
		face_descriptor = [f for f in face_descriptor] #converts to list
		face_descriptor = np.asarray(face_descriptor, dtype=np.float64)
		face_descriptor = face_descriptor[np.newaxis, :]
		
		#print(face_descriptor.shape)
		if(face_descriptors is None):
			face_descriptors = face_descriptor
		else:
			face_descriptors = np.concatenate((face_descriptors,face_descriptor),axis = 0)
		index[idx] = path
		idx += 1
	
#print(index)

print("Distance, same image:",np.linalg.norm(face_descriptors[131] - face_descriptors[131]))
print("Distance, same person, different image:",np.linalg.norm(face_descriptors[131] - face_descriptors[129]))
print("Distance, different people, same expression:",np.linalg.norm(face_descriptors[93] - face_descriptors[129]))
print("Distance, different people, different expression:",np.linalg.norm(face_descriptors[94] - face_descriptors[129]))

print("Compare 131 to all faces:",np.linalg.norm(face_descriptors[131] - face_descriptors, axis=1))
best_match = np.argmin(np.linalg.norm(face_descriptors[0] - face_descriptors[1:], axis=1))
print("Best match for ",index[0],"is",index[best_match])


threshold = 0.5
predictions = []
expected_outputs = []

paths = [os.path.join("Datasets/yalefaces/test",f) for f in os.listdir("Datasets/yalefaces/test")]
for path in paths:
	image = Image.open(path).convert("RGB")
	image_np = np.array(image,"uint8")
	face_detection = face_detector(image_np, 1)
	for face in face_detection:
		points = points_detector(image_np,face)
		face_descriptor = face_descriptor_extractor.compute_face_descriptor(image_np,points)
		face_descriptor = [f for f in face_descriptor] #converts to list
		face_descriptor = np.asarray(face_descriptor, dtype=np.float64)
		face_descriptor = face_descriptor[np.newaxis, :]
		
		distances = np.linalg.norm(face_descriptor-face_descriptors, axis=1)
		min_index = np.argmin(distances)
		min_distance = distances[min_index]
		if min_distance <= threshold:
			name_pred = int(os.path.split(index[min_index])[1].split(".")[0].replace("subject",""))
		else:
			name_pred = "Not identified"
		name_real = int(os.path.split(path)[1].split(".")[0].replace("subject",""))
		predictions.append(name_pred)
		expected_outputs.append(name_real)
		
		cv2.putText(image_np, "Pred: " + str(name_pred), (10,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))
		cv2.putText(image_np, "Real: " + str(name_real), (10,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))
		#cv2.imshow("",image_np)
		#cv2.waitKey(0)
#print(face_descriptors.shape)
			
predictions = np.array(predictions)
expected_outputs = np.array(expected_outputs)

print(predictions)
print(expected_outputs)
		
from sklearn.metrics import accuracy_score
print(accuracy_score(expected_outputs, predictions))

	