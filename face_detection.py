import cv2

image = cv2.imread("Images/people1.jpg")
print(image.shape)

#scale image`
image = cv2.resize(image,(int(image.shape[1]*0.5),int(image.shape[0]*0.5)))
print(image.shape)

#convert to grayscale
image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

cv2.imshow("",image_gray)
cv2.waitKey(0)
