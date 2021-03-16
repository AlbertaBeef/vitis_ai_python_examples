'''
Copyright 2021 Avnet Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

# USAGE
# python face_landmarks_dlib.py [--input 0] [--detthreshold 0.55] [--nmsthreshold 0.35]

from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import pathlib
import xir
import os
import math
import threading
import time
import sys
import argparse

from imutils.video import FPS

sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('./'))
from vitis_ai_vart.facedetect import FaceDetect
from vitis_ai_vart.facelandmark import FaceLandmark
from vitis_ai_vart.utils import get_child_subgraph_dpu

import dlib

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=False,
	help = "input camera identifier (default = 0)")
ap.add_argument("-d", "--detthreshold", required=False,
	help = "face detector softmax threshold (default = 0.55)")
ap.add_argument("-n", "--nmsthreshold", required=False,
	help = "face detector NMS threshold (default = 0.35)")
args = vars(ap.parse_args())

if not args.get("input",False):
  inputId = 0
else:
  inputId = int(args["input"])
print('[INFO] input camera identifier = ',inputId)

if not args.get("detthreshold",False):
  detThreshold = 0.55
else:
  detThreshold = float(args["detthreshold"])
print('[INFO] face detector - softmax threshold = ',detThreshold)

if not args.get("nmsthreshold",False):
  nmsThreshold = 0.35
else:
  nmsThreshold = float(args["nmsthreshold"])
print('[INFO] face detector - NMS threshold = ',nmsThreshold)

# Initialize Vitis-AI/DPU based face detector
densebox_xmodel = "/usr/share/vitis_ai_library/models/densebox_640_360/densebox_640_360.xmodel"
densebox_graph = xir.Graph.deserialize(densebox_xmodel)
densebox_subgraphs = get_child_subgraph_dpu(densebox_graph)
assert len(densebox_subgraphs) == 1 # only one DPU kernel
densebox_dpu = vart.Runner.create_runner(densebox_subgraphs[0],"run")
dpu_face_detector = FaceDetect(densebox_dpu,detThreshold,nmsThreshold)
dpu_face_detector.start()

# Initialize Vitis-AI/DPU based face landmark
landmark_xmodel = "/usr/share/vitis_ai_library/models/face_landmark/face_landmark.xmodel"
landmark_graph = xir.Graph.deserialize(landmark_xmodel)
landmark_subgraphs = get_child_subgraph_dpu(landmark_graph)
assert len(landmark_subgraphs) == 1 # only one DPU kernel
landmark_dpu = vart.Runner.create_runner(landmark_subgraphs[0],"run")
dpu_face_landmark = FaceLandmark(landmark_dpu)
dpu_face_landmark.start()

# Initialize DLIB based face detector
dlib_face_detector = dlib.get_frontal_face_detector()

# Initialize DLIB based face landmark
dlib_landmark_model = "./models/shape_predictor_68_face_landmarks.dat"
dlib_face_landmark = dlib.shape_predictor(dlib_landmark_model)

# algorithm selection
use_dlib_detection = False
use_dlib_landmarks = True
print("[INFO] face detection = VART")
print("[INFO] face landmarks = DLIB")

# Initialize the camera input
print("[INFO] starting camera input ...")
cam = cv2.VideoCapture(inputId)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
if not (cam.isOpened()):
    print("[ERROR] Failed to open camera ", inputId )
    exit()

# Camera internals
ret,frame = cam.read()
size=frame.shape

# start the FPS counter
fps = FPS().start()

# init the real-time FPS display
rt_fps_count = 0;
rt_fps_time = cv2.getTickCount()
rt_fps_valid = False
rt_fps = 0.0
rt_fps_message = "FPS: {0:.2f}".format(rt_fps)
rt_fps_x = 10
rt_fps_y = size[0]-10

# loop over the frames from the video stream
while True:
	# Update the real-time FPS counter
	if rt_fps_count == 0:
		rt_fps_time = cv2.getTickCount()

	# Capture image from camera
	ret,frame = cam.read()
	dlib_image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

	faces = []
    
	if use_dlib_detection == False:
		# Vitis-AI/DPU based face detector
		faces = dpu_face_detector.process(frame)
		#print(faces)
	
	if use_dlib_detection == True:
		# DLIB based face detector
		dlib_faces = dlib_face_detector(dlib_image, 0)
		for face in dlib_faces:
			faces.append( (face.left(),face.top(),face.right(),face.bottom()) )
		#print(faces)
      
	# loop over the faces
	for i,(left,top,right,bottom) in enumerate(faces): 

		# draw a bounding box surrounding the object so we can
		# visualize it
		cv2.rectangle( frame, (left,top), (right,bottom), (0,255,0), 2)

		# extract the face ROI
		startX = int(left)
		startY = int(top)
		endX   = int(right)
		endY   = int(bottom)
		#print( startX, endX, startY, endY )
		widthX   = endX-startX
		heightY  = endY-startY
		face = frame[startY:endY, startX:endX]

		if use_dlib_landmarks == False:

			# extract face landmarks
			landmarks = dpu_face_landmark.process(face)

                        # calculate coordinates for full frame
			for i in range(5):
                            landmarks[i,0] = startX + landmarks[i,0]*widthX
                            landmarks[i,1] = startY + landmarks[i,1]*heightY

			# draw landmarks
			for i in range(5):
                            x = int(landmarks[i,0])
                            y = int(landmarks[i,1])
                            cv2.circle( frame, (x,y), 3, (255,255,255), 2)
       

		if use_dlib_landmarks == True:

			# extract face landmarks with DLIB
			dlib_rect = dlib.rectangle( startX,startY,endX,endY )
			dlib_landmarks = dlib_face_landmark(dlib_image,dlib_rect)

			# draw landmarks
			for i in range(dlib_landmarks.num_parts):
                            x = int(dlib_landmarks.part(i).x)
                            y = int(dlib_landmarks.part(i).y)
                            cv2.circle( frame, (x,y), 3, (255,255,255), 2)


	# Display Status
	status = "Status :"
	status = status + " FaceDetect="
	if use_dlib_detection == True:
		status = status + "DLIB"
	else:
		status = status + "VART"
	status = status + " Landmark="
	if use_dlib_landmarks == True:
		status = status + "DLIB"
	else:
		status = status + "VART"
	if rt_fps_valid == True:
		status = status + " " + rt_fps_message
	cv2.putText(frame, status, (rt_fps_x,rt_fps_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

	# Display the processed image
	cv2.imshow("Face Detection with Landmarks", frame)
	key = cv2.waitKey(1) & 0xFF

	# Update the FPS counter
	fps.update()

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# if the `d` key was pressed, toggle between detection algorithms
	if key == ord("d"):
		use_dlib_detection = not use_dlib_detection
		if use_dlib_detection == True:
		   print("[INFO] face detection = DLIB")
		else:
		   print("[INFO] face detection = VART")

		
	# if the `l` key was pressed, toggle between landmark algorithms
	if key == ord("l"):
		use_dlib_landmarks = not use_dlib_landmarks
		if use_dlib_landmarks == True:
		   print("[INFO] face landmarks = DLIB")
		else:
		   print("[INFO] face landmarks = VART")


	# Update the real-time FPS counter
	rt_fps_count = rt_fps_count + 1
	if rt_fps_count >= 10:
		t = (cv2.getTickCount() - rt_fps_time)/cv2.getTickFrequency()
		rt_fps_valid = True
		rt_fps = 10.0/t
		rt_fps_message = "FPS: {0:.2f}".format(rt_fps)
		#print("[INFO] ",rt_fps_message)
		rt_fps_count = 0


# Stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] elapsed FPS: {:.2f}".format(fps.fps()))

# Stop the face detector
dpu_face_detector.stop()
del densebox_dpu
dpu_face_landmark.stop()
del landmark_dpu

# Cleanup
cv2.destroyAllWindows()
