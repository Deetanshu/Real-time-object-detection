# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 10:21:49 2018

@author: deept
"""

import cv2 as cv
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np


def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err
vs=VideoStream(src=0).start()
fps=FPS().start()
while True:
    frame=vs.read()
    while True:
        frame1=vs.read()
        m = mse(frame1, frame)
        if m>0:
            break
    cv.imshow("FRAME: ",frame1)
    key = cv.waitKey(1)&0xFF
    if key == ord("q"):
        break
    fps.update()

#cleanup
fps.stop()
print("Elapsed time: {:.2f}".format(fps.elapsed()))
print("Approx FPS: {:.2f}".format(fps.fps()))

cv.destroyAllWindows()
vs.stop()    