import cv2

image = cv2.imread("Images/people1.jpg")
#image = cv2.resize(image,(800,600))

#convert to grayscale
image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

eye_detector = cv2.CascadeClassifier("Cascades/haarcascade_eye.xml")
face_detector = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")

face_detections = face_detector.detectMultiScale(image_gray, scaleFactor=1.3,minSize=(40,40))
eye_detections = eye_detector.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=10,maxSize=(50,50))

#draw rectangles around faces
for(x,y,w,h) in face_detections:
	cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)

#draw rectangles around eyes
for(x,y,w,h) in eye_detections:
	cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)
	
cv2.imshow("",image)
cv2.waitKey(0)