import cv2

choice = input("Press 1 to use KCF tracker. Press 2 for CSRT. ")

if choice == "1":
	print("Using KCF tracker. It is fast, but less accurate than CSRT.")
	tracker = cv2.TrackerKCF_create()
else:
	print("Using CSRT tracker. It is slower, but more accurate")
	#CSRT is overall better for street scene, because there are more objects
	# and they move faster. KCF is easily confused.
	tracker = cv2.TrackerCSRT_create()

filename = "Videos/race.mp4"
choice = input("Press 1 to use race video. Press 2 for street video. ")

if choice == "2":
	filename = "Videos/street.mp4"

video = cv2.VideoCapture(filename)
ok, frame = video.read() #reads first frame and determines if it is ok
bbox = cv2.selectROI(frame)
#print(bbox)

ok = tracker.init(frame,bbox)
#print(ok)

while True:
	ok, frame = video.read()
	if not ok:
		break
	ok, bbox = tracker.update(frame)
	#print(bbox)
	
	if ok:
		(x, y, w, h) = [int(v) for v in bbox]
		cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0), 2,1)
	else:
		cv2.putText(frame, 'Error', (100,80),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
		
	cv2.imshow("Tracking", frame)
	if cv2.waitKey(1) & 0XFF == 27: # ESC key
		break 