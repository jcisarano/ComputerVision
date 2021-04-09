import cv2

image = cv2.imread("Images/people3.jpg")
#image = cv2.resize(image,(800,600))

#convert to grayscale
image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

body_detector = cv2.CascadeClassifier("Cascades/fullbody.xml")

body_detections = body_detector.detectMultiScale(image_gray)
#body_detections = body_detector.detectMultiScale(image_gray, scaleFactor=1.3,minSize=(40,40))

#draw rectangles around detections
for(x,y,w,h) in body_detections:
	cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
	
cv2.imshow("",image)
cv2.waitKey(0)