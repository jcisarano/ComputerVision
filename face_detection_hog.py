import dlib
import cv2

image = cv2.imread("Images/people2.jpg")

face_detector_hog = dlib.get_frontal_face_detector()

detections = face_detector_hog(image, 1)
print(detections)
print("Found",len(detections),"faces")

for face in detections:
	l,t,r,b = face.left(), face.top(), face.right(), face.bottom()
	cv2.rectangle(image, (l,t), (r,b), (0,255,0), 2)

cv2.imshow("",image)
cv2.waitKey(0)