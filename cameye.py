# USAGE
# python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import urllib.request as urlp 
import pickle
from FingerDetection import manage_image_opr
from gtts import gTTS 
import pygame
import serial

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=False,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3-tiny.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3-tiny.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
writer = None
(W, H) = (None, None)
knws = np.zeros(80)
knws[72] = 26.5
knws[68] = 27.5
knws[69] = 27
"""
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1
"""
# loop over frames from the video file stream
url='http://192.168.1.2:8080/shot.jpg'
hand_hist = pickle.load( open( "save.p", "rb" ) )
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
ser.flush()
buffer_string = ''
data1 = 10
while True:
	pygame.mixer.init()
	if ser.in_waiting > 0:
		data = ser.read(ser.inWaiting())
		sr = data.decode()
		buffer_string = buffer_string + sr
		if '\n' in buffer_string:
			lines = buffer_string.split('\n')
			last_received = lines[-2]
			data1 = int(last_received)
			#print(data1)
		if(data1<5):
			#print("stop walking")
			pygame.mixer.music.load("text1.mp3")
			pygame.mixer.music.play()
			while pygame.mixer.music.get_busy() == True:
				continue
			
	# read the next frame from the file
	imgResp = urlp.urlopen(url)
	imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
	# Finally decode the array to OpenCV usable format ;) 
	frame = cv2.imdecode(imgNp,-1)
	grabbed = 1
	pnt_f=manage_image_opr(frame, hand_hist)
	#(grabbed, frame) = vs.read()
	far_point = pnt_f
	#far_point = None
	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > args["confidence"]:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				"""
				if(classID == 0):
					xmin = max(x-20,0)
					ymin = max(y-20,0)
					xmax = min(x+int(width)+20,480)
					ymax = min(y+int(height)+20,640)
					patch = frame[ymin:ymax,xmin:xmax]
					far_point=manage_image_opr(patch, hand_hist)
					if(far_point!=None):
						#print("farpoint1"+str(far_point))
						dx = far_point[0]+xmin
						dy = far_point[1]+ymin
						pnt_f = (dx,dy)
						#print("farpoint2"+str(pnt))
						cv2.circle(frame, pnt_f, 5, [0, 0, 255], -1)
					#cv2.imshow('output',patch)
					"""
				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			if(1):
			# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])
				pnt = (x,y)
				pnt1 =  (x+w,y+h)
				# draw a bounding box rectangle and label on the frame
				color = [int(c) for c in COLORS[classIDs[i]]]
				if(far_point!=None):
					x1= (pnt_f[0]-x)
					y1 = (pnt_f[1]-y)
					x2 = (x+w-pnt_f[0])
					y2 = (y+h-pnt_f[1])
					if(x1>0 and y1>0 and x2>0 and y2>0):
						#print("xy : " + str(pnt) + ", later :"+str(pnt1)+", farthest Point : " + str(far_point))
						cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
						#text = "{}: {:.4f}".format(LABELS[classIDs[i]],
						#		confidences[i])
						dist = knws[classIDs[i]]*562/w
						text = "{} at {:.1f}in".format(LABELS[classIDs[i]],dist)
						speech = gTTS(text = text, lang = 'en', slow = False)
						speech.save("text.mp3")
						pygame.mixer.music.load("text.mp3")
						pygame.mixer.music.play()
						while pygame.mixer.music.get_busy() == True:
							continue
						cv2.putText(frame, text, (x, y - 5),
								cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		# some information on processing single frame
		"""
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))
		"""

	# write the output frame to disk
	cv2.imshow('output1',frame)
	cv2.waitKey(14)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	writer.write(frame)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
#vs.release()
