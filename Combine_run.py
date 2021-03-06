# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 23:33:37 2018

@author: deept
"""
#imports
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2 as cv
#import matplotlib.pyplot as plt
import keras

#Setting path to tensorflow model files: 
prototxt="Model/export_model.pbtxt"
cmodel="Model/frozen_inference_graph.pb"

#Argument Parser. Edit: Added Android streaming as an argument as well.
ap=argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="Minimum probability")
ap.add_argument("-a", "--android", type=int, default=0, help="0 is Webcam and 1 is android cam")
args=vars(ap.parse_args())

#Load pre-trained model:
print("Loading Model: ")
net = cv.dnn.readNetFromTensorflow(cmodel, prototxt)
swapRB = True

#Create list of classes and color set:
classNames = { 0: 'background',
            1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
            7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
            13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
            18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
            24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
            32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
            37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
            41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
            46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
            51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
            56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
            61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
            67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
            75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
            80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
            86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush' }
colors=np.random.uniform(0,255, size=(91, 3))
choice=args["android"]

#Start Video Stream if arg is 0.
if choice==1:
    from urllib.request import urlopen
    diff=(1280/300)
    osize=1280
    url="http://"+input("Please enter the URL of the android device: ")+"/shot.jpg"
    print("Connecting to ",url)
    fps=FPS().start()
else:
    from imutils.video import VideoStream
    print("Starting video stream:")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    fps=FPS().start()
    diff=(480/300)
    osize=640
#Loop for frames:
while True:
    if choice==1:
        imgResp = urlopen(url)
        imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
        img = cv.imdecode(imgNp,-1)
        frame=img
        original=frame
    else:
        frame=vs.read()
        while frame is None:
            print("Fetching frame again.")
            frame=vs.read(),
        original=cv.flip(frame,1)
    frame=imutils.resize(frame, width=400)
    (h,w)=frame.shape[:2]
    blob=cv.dnn.blobFromImage(cv.resize(frame, (300,300)), 0.007843, (300,300), 127.5, swapRB=True, crop=False)
    blob = cv.dnn.blobFromImage(frame, 0.007843, (300,300), (127.5,127.5,127.5), swapRB)
    net.setInput(blob)
    detections=net.forward()
    for i in np.arange(0, detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence > args["confidence"]:
            class_id = int(detections[0, 0, i, 1])
            box=detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX, startY, endX, endY)=box.astype("int")
            if class_id in classNames:
                label = classNames[class_id] + ": " + str(confidence)
                box=detections[0,0,i,3:7]*np.array([w,h,w,h])
                (startX, startY, endX, endY)=box.astype("int")
                if choice==1:
                    cv.rectangle(original, (int(startX*diff), int(startY*diff)), (int(endX*diff), int(endY*diff)), colors[class_id], 1)
                    y=startY - 15 if startY - 15 > 15 else startY + 15
                    cv.putText(original, label, (int(startX*diff),int(y*diff)), cv.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id], 1)
                else:
                    cv.rectangle(original, (640-int(startX*diff), int(startY*diff)), (640-int(endX*diff), int(endY*diff)), colors[class_id], 1)
                    y=startY - 15 if startY - 15 > 15 else startY + 15
                    cv.putText(original, label, (640-int(endX*diff),int(y*diff)), cv.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id], 1)
    #show images, within the bigger loop:
    cv.imshow("Original", original)
    key = cv.waitKey(1)&0xFF
    if key == ord("q"):
        break
    fps.update()

#cleanup
fps.stop()
print("Elapsed time: {:.2f}".format(fps.elapsed()))
print("Approx FPS: {:.2f}".format(fps.fps()))

if choice==1:
    print("Connection Terminated")
else:
    vs.stop()
cv.destroyAllWindows()
exit()