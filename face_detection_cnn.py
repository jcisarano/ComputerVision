import dlib
import cv2

image = cv2.imread("Images/people2.jpg")

face_detector = dlib.cnn_face_detection_model_v1("Weights/mmod_human_face_detector.dat")

detections = face_detector(image, 1)
print(detections)
print("Found",len(detections),"faces")

for face in detections:
	l,t,r,b,c = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom(), face.confidence
	print(c)
	cv2.rectangle(image, (l,t), (r,b), (0,255,0), 2)

cv2.imshow("",image)
cv2.waitKey(0)
