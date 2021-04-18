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
lbph_classifier.write("lbph_classifier.yml") #saves classifier to disk