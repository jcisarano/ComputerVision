import cv2

image = cv2.imread("Images/people1.jpg")
print(image.shape)

#scale image`
image = cv2.resize(image,(int(image.shape[1]*0.5),int(image.shape[0]*0.5)))
print(image.shape)

#convert to grayscale
image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#cv2.imshow("",image_gray)
#cv2.waitKey(0)

face_detector = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")
detections = face_detector.detectMultiScale(image_gray)
#detections are positions of faces (x,y,width,height)
print(detections)
print(len(detections))

#draw rectangles around faces
for(x,y,w,h) in detections:
	#print(x,y,w,h)
	cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
	

cv2.imshow("",image)
cv2.waitKey(0)
