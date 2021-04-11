import cv2
import dlib

image = cv2.imread("Images/people3.jpg")
print(image.shape)

#convert to grayscale
image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

face_detector = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")
detections = face_detector.detectMultiScale(image_gray, scaleFactor=1.001, minNeighbors=5,minSize=(5,5))

#draw rectangles around faces
for(x,y,w,h) in detections:
	#print(x,y,w,h)
	cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)

cv2.imshow("",image)
cv2.waitKey(0)


print("Start HOG")
image = cv2.imread("Images/people3.jpg")

face_detector_hog = dlib.get_frontal_face_detector()

detections = face_detector_hog(image, 4)
print(detections)
print("Found",len(detections),"faces")

for face in detections:
	l,t,r,b = face.left(), face.top(), face.right(), face.bottom()
	cv2.rectangle(image, (l,t), (r,b), (0,255,255), 2)

cv2.imshow("",image)
cv2.waitKey(0)

print("Start CNN")
image = cv2.imread("Images/people3.jpg")

face_detector = dlib.cnn_face_detection_model_v1("Weights/mmod_human_face_detector.dat")

print("CNN start detections")
detections = face_detector(image, 1)
print(detections)
print("Found",len(detections),"faces")

for face in detections:
	l,t,r,b,c = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom(), face.confidence
	print(c)
	cv2.rectangle(image, (l,t), (r,b), (0,255,0), 2)

cv2.imshow("",image)
cv2.waitKey(0)