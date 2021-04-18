from PIL import Image
import cv2
import numpy as np

#import zipfile
#path = "Datasets/yalefaces.zip"
#zip_object = zipfile.ZipFile(file=path,mode='r')
#zip_object.extractall("./")
#zip_object.close()

import os
#print(os.listdir("Datasets/yalefaces/train"))

def get_image_data():
	paths = [os.path.join("Datasets/yalefaces/train",f) for f in os.listdir("Datasets/yalefaces/train")]
	#print(paths)
	faces = []
	ids = []
	for path in paths:
		image = Image.open(path).convert("L") #L mode is single-channnel image interpreted as grayscale
		#print(type(image))
		image_np = np.array(image,"uint8") #convert image to numpy format
		id = int(os.path.split(path)[1].split(".")[0].replace("subject","")) #extract id num from file path
		#print(id)
		ids.append(id)
		faces.append(image_np)
	
	return np.array(ids), faces
	
ids, faces = get_image_data()

#print(len(ids))
#print(len(faces))
#print(faces[0].shape)

#defaults to 8x8 grid, so 64 histograms per image
lbph_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_classifier.train(faces,ids)
lbph_classifier.write("Classifiers/lbph_classifier.yml") #saves classifier to disk

lbph_face_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_face_classifier.read("Classifiers/lbph_classifier.yml")

test_image = "Datasets/yalefaces/test/subject10.sad.gif"
image = Image.open(test_image).convert("L")
image_np = np.array(image,"uint8")
prediction = lbph_face_classifier.predict(image_np)
#print(prediction)

expected_output = int(os.path.split(test_image)[1].split(".")[0].replace("subject",""))
#print(expected_output)
print("Expected subject:",expected_output,"Predicted:",prediction[0])

paths = [os.path.join("Datasets/yalefaces/test",f) for f in os.listdir("Datasets/yalefaces/test")]
predictions = []
expected_outputs = []
for path in paths:
	#print(path)
	image = Image.open(path).convert("L")
	image_np = np.array(image,"uint8")
	prediction, _ = lbph_face_classifier.predict(image_np)
	expected_output = int(os.path.split(path)[1].split(".")[0].replace("subject",""))
	
	predictions.append(prediction)
	expected_outputs.append(expected_output)
	
predictions = np.array(predictions)
expected_outputs = np.array(expected_outputs)

print(predictions)
print(expected_outputs)

from sklearn.metrics import accuracy_score
print(accuracy_score(expected_outputs,predictions) )

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(expected_outputs,predictions)
print(cm)

import seaborn
import matplotlib.pyplot as plt
seaborn.heatmap(cm,annot=True)
plt.show()
