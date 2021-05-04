import cv2

tracker = cv2.TrackerKCF_create()
video = cv2.VideoCapture("Videos/race.mp4")
ok, frame = video.read() #reads first frame and determines if it is ok
bbox = cv2.selectROI(frame)
print(bbox)

ok = tracker.init(frame,bbox)
print(ok)

while True:
	ok, frame = video.read()