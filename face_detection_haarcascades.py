import cv2

image = cv2.imread("Images/people2.jpg")
print(image.shape)

#scale image
#image = cv2.resize(image,(int(image.shape[1]*0.5),int(image.shape[0]*0.5)))
#image = cv2.resize(image,(800,600))
#print(image.shape)

#convert to grayscale
image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#cv2.imshow("",image_gray)
#cv2.waitKey(0)

face_detector = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")

#detections are positions of faces (x,y,width,height)
#scale factor can be used to allow for different image size and size of expected faces
#minNeighbors is number of nearby candidate rectangles used to generate the final bounding box
#this can help eliminate false positives
#minSize is min bounding box size, default = (30,30)
#maxSize is max bounding box size
detections = face_detector.detectMultiScale(image_gray, scaleFactor=1.2, minNeighbors=3,minSize=(40,40),maxSize=(100,100))
print(detections)
print(len(detections))

#draw rectangles around faces
for(x,y,w,h) in detections:
	#print(x,y,w,h)
	cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
	

cv2.imshow("",image)
cv2.waitKey(0)
