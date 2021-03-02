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
# python face_headpose.py [--input 0] [--detthreshold 0.55] [--nmsthreshold 0.35]

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

# Initialize the camera input
print("[INFO] starting camera input ...")
cam = cv2.VideoCapture(inputId)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
if not (cam.isOpened()):
    print("[ERROR] Failed to open camera ", inputId )
    exit()

# 3D model points.
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        
                        ])

# Camera internals
ret,frame = cam.read()
size=frame.shape
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
print("[INFO] Camera Matrix :\n {0}".format(camera_matrix));

# start the FPS counter
fps = FPS().start()

# loop over the frames from the video stream
while True:
	# Capture image from camera
	ret,frame = cam.read()

	# Vitis-AI/DPU based face detector
	faces = dpu_face_detector.process(frame)
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
		face = frame[startY:endY, startX:endX]

		# extract face landmarks
		landmarks = dpu_face_landmark.process(face)

		# draw landmarks
		#for i in range(5):
		#	x = int(landmarks[i,0] * (endX-startX))
		#	y = int(landmarks[i,1] * (endY-startY))
		#	cv2.circle( face, (x,y), 3, (255,255,255), 2)
                        
		widthX   = endX-startX
		heightY  = endY-startY

		# prepare 2D points
		image_points = np.array([
                            (landmarks[2,0]*widthX, landmarks[2,1]*heightY), # Nose tip
                            (landmarks[2,0]*widthX, landmarks[2,1]*heightY), # Chin (place-holder for now)
                            (landmarks[0,0]*widthX, landmarks[0,1]*heightY), # Left eye left corner
                            (landmarks[1,0]*widthX, landmarks[1,1]*heightY), # Right eye right corne
                            (landmarks[3,0]*widthX, landmarks[3,1]*heightY), # Left Mouth corner
                            (landmarks[4,0]*widthX, landmarks[4,1]*heightY)  # Right mouth corner
                        ], dtype="double")

		# estimate approximate location of chin
		# let's assume that the chin location will behave similar as the nose location
		eye_center_x = (image_points[2][0] + image_points[3][0])/2;
		eye_center_y = (image_points[2][1] + image_points[3][1])/2;
		nose_offset_x = (image_points[0][0] - eye_center_x);
		nose_offset_y = (image_points[0][1] - eye_center_y);
		mouth_center_x = (image_points[4][0] + image_points[5][0])/2;
		mouth_center_y = (image_points[4][1] + image_points[5][1])/2;
		image_points[1] = (mouth_center_x + nose_offset_x, mouth_center_y + nose_offset_y);

		# calculate head pose
		dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
		(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

		#print "Rotation Vector:\n {0}".format(rotation_vector)
		#print "Translation Vector:\n {0}".format(translation_vector)

  
		# Project a 3D point (0, 0, 1000.0) onto the image plane.
		# We use this to draw a line sticking out of the nose

		(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

		for p in image_points:
		  #cv2.circle(face, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
		  cv2.circle(face, (int(p[0]), int(p[1])), 3, (255,255,255), 2)


		#p1 = ( int(image_points[0][0]), int(image_points[0][1]))
		#p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
		#cv2.line(face, p1, p2, (255,0,0), 2)

		# draw head pose vector (draw in original image for full vector length)
		p1 = ( startX+int(image_points[0][0]), startY+int(image_points[0][1]))
		p2 = ( startX+int(nose_end_point2D[0][0][0]), startY+int(nose_end_point2D[0][0][1]))
		cv2.line(frame, p1, p2, (255,0,0), 2)

	# Display the processed image
	cv2.imshow("Face Detection with Landmarks", frame)
	key = cv2.waitKey(1) & 0xFF

	# Update the FPS counter
	fps.update()

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

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
