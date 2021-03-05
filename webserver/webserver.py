from flask import Flask,render_template,request,Response
import numpy as np
import cv2
import base64
import sys
import json
import os
import vart
import pathlib
import xir


sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('./'))
from vitis_ai_vart.facedetect import FaceDetect
from vitis_ai_vart.facelandmark import FaceLandmark
from vitis_ai_vart.utils import get_child_subgraph_dpu


# Define App
app = Flask(__name__,template_folder="templates")

# The home page is routed to index.html inside
@app.route('/')
def index():
   return render_template('index.html')

# Load Digit Recogniztion model
#net = cv2.dnn.readNetFromONNX('model.onnx')

# Implements softmax function
#def softmax(x):
#    """Compute softmax values for each sets of scores in x."""
#    return np.exp(x) / np.sum(np.exp(x), axis=0)

# Vitis-AI/DPU variables
global detThreshold
global nmsThreshold
global ai_algorithm
detThreshold = 0.55
nmsThreshold = 0.35
#ai_algorithm = 0 # none
ai_algorithm = 1 # face detection


# Handles uploaded image
@app.route('/upload',methods=["POST"])
def upload():
  # Get uploaded form
  d = request.form
  # Extract the data field
  data = d.get('data')
  
  # The first part of the string simply indicates 
  # what kind of file it is. So we extract only the data part. 
  data = data.split(',')[1]

  # Get base64 decoded 
  data = base64.decodebytes(data.encode())
  
  # Convert to numpy array
  nparr = np.frombuffer(data, np.uint8)

  # Read image
  img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
  cv2.imwrite("/tmp/test.jpg", img)
  
  # Create a 4D blob from image
  blob = cv2.dnn.blobFromImage(img, 1/255, (28, 28))

  # Run a model
  #net.setInput(blob)
  #out = net.forward()
  
  # Get a class with a highest score
  #out = softmax(out.flatten())
  #classId = np.argmax(out)
  #confidence = out[classId]
  classId = 1
  confidence = 0.00

  # Print results on the server side
  print("classId: {} confidence: {}".format(classId, confidence), file=sys.stdout)
  
  # Return result as a json object
  return json.dumps({'success':True, 'class': int(classId), 'confidence': float(confidence)}), 200, {'ContentType':'application/json'} 

@app.route('/set_threshold/<slider>/<value>',methods=["POST"])
def set_threshold(slider,value):
   global detThreshold
   global nmsThreshold

   if slider == "det":
      detThreshold = (float(value)/100.0)
      print("[INFO] detThreshold = ",detThreshold)

   if slider == "nms":
      nmsThreshold = (float(value)/100.0)
      print("[INFO] nmsThreshold = ",nmsThreshold)

   # Return result as a json object
   return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 

@app.route('/set_algorithm/<algo>',methods=["POST"])
def set_algorithm(algo):
   global ai_algorithm

   if algo == "none": 
      ai_algorithm = 0
      print("[INFO] algorithm = none")

   if algo == "faces": 
      ai_algorithm = 1
      print("[INFO] algorithm = face detection")

   if algo == "landmarks": 
      ai_algorithm = 2
      print("[INFO] algorithm = face landmarks")

   if algo == "headpose": 
      ai_algorithm = 3
      print("[INFO] algorithm = head pose estimation")

   # Return result as a json object
   return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 
   


def generate():
   global detThreshold
   global nmsThreshold
   global ai_algorithm

   print("[INFO] Vitis-AI/DPU based face detector initialization ...")
   densebox_xmodel = "/usr/share/vitis_ai_library/models/densebox_640_360/densebox_640_360.xmodel"
   densebox_graph = xir.Graph.deserialize(densebox_xmodel)
   densebox_subgraphs = get_child_subgraph_dpu(densebox_graph)
   assert len(densebox_subgraphs) == 1 # only one DPU kernel
   densebox_dpu = vart.Runner.create_runner(densebox_subgraphs[0],"run")
   dpu_face_detector = FaceDetect(densebox_dpu,detThreshold,nmsThreshold)
   dpu_face_detector.start()

   print("[INFO] Vitis-AI/DPU based face landmark initialization ...")
   landmark_xmodel = "/usr/share/vitis_ai_library/models/face_landmark/face_landmark.xmodel"
   landmark_graph = xir.Graph.deserialize(landmark_xmodel)
   landmark_subgraphs = get_child_subgraph_dpu(landmark_graph)
   assert len(landmark_subgraphs) == 1 # only one DPU kernel
   landmark_dpu = vart.Runner.create_runner(landmark_subgraphs[0],"run")
   dpu_face_landmark = FaceLandmark(landmark_dpu)
   dpu_face_landmark.start()

   print("[INFO] WEBCAM (/dev/video0) openned by generate() function.")
   cap = cv2.VideoCapture(0)
   cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

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
   ret,frame = cap.read()
   size=frame.shape
   focal_length = size[1]
   center = (size[1]/2, size[0]/2)
   camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
   print("[INFO] Camera Matrix :\n {0}".format(camera_matrix));

   while True:
      # if closed by someone else
      #if not cap.isOpened():
      #   print("[INFO] WEBCAM (/dev/video0) released by someone else.")
      #   break

      # Capture frame-by-frame
      ret, frame = cap.read()
      if not ret:
         print("[ERROR] Error capturing video")
         print("[INFO] WEBCAM (/dev/video0) released.")
         cap.release()
         break

      # Processing
      if ai_algorithm > 0:
         # Vitis-AI/DPU based face detector
         dpu_face_detector.config( detThreshold, nmsThreshold )
         faces = dpu_face_detector.process(frame)
      
         # loop over the faces
         for i,(left,top,right,bottom) in enumerate(faces): 
     
            # draw a bounding box surrounding the object so we can visualize it
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

            if ai_algorithm == 2:
               # draw landmarks
               for i in range(5):
                  x = int(landmarks[i,0] * (endX-startX))
                  y = int(landmarks[i,1] * (endY-startY))
                  cv2.circle( face, (x,y), 3, (255,255,255), 2)
                        
            if ai_algorithm == 3:
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

               # Project a 3D point (0, 0, 1000.0) onto the image plane.
               # We use this to draw a line sticking out of the nose
               (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

               for p in image_points:
                  #cv2.circle(face, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
                  cv2.circle(face, (int(p[0]), int(p[1])), 3, (255,255,255), 2)

               # draw head pose vector (draw in original image for full vector length)
               p1 = ( startX+int(image_points[0][0]), startY+int(image_points[0][1]))
               p2 = ( startX+int(nose_end_point2D[0][0][0]), startY+int(nose_end_point2D[0][0][1]))
               cv2.line(frame, p1, p2, (255,0,0), 2)

      # Encode video frame to JPG
      (flag, encodedImage) = cv2.imencode(".jpg", frame)
      if not flag:
         continue
      
      # yield the output frame in the byte format
      yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
   return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
   #app.run(debug = True)
   app.run(host='0.0.0.0', port=80, debug=True)
