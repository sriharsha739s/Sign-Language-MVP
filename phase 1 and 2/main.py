import cv2
import time
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch

##USING YOLO TO RUN


def preprocess(frame):
	f = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	f = cv2.GaussianBlur(f, (5,5), 0)
	return f

def segment(img):
	_, thresholded = cv2.threshold(img, 25,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	contours,hierarchy = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	if len(contours) == 0:
		return
	else:
		#find max contour which is the hand
		cnt = max(contours, key = lambda x: cv2.contourArea(x))
		return (thresholded,cnt)


model = tf.keras.models.load_model('modelTest2.h5')


def make_prediction(frame):
	(h,w) = frame.shape[:2]
	final_array=cv2.resize(frame,(50,50))
	final_array = np.reshape(final_array, (1,50,50))
	# final_array = np.repeat(final_array[..., np.newaxis], 3, -1)
	final_array = np.resize(final_array,(50,50,1))
	final_array=np.expand_dims(final_array,axis=0)
	predictions = model.predict(final_array)
	predicted = np.argmax(predictions)
	return predicted

cap = cv2.VideoCapture(0)
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Best of Luck', 'You', 'I/Me', 'Like', 'Remember ', 'Love', 'Fuck', 'I love you']


model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
while True:
	success,frame = cap.read()
	if not success:
		print("Error reading")
		break

	frame = cv2.flip(frame,1)

	clone = frame.copy()
	cv2.rectangle(frame, (500,500), (50,50), (0,255,0),0)
	cropped_img = clone[50:500, 50:500]

	img = preprocess(cropped_img)


	thresholded, maxContour = segment(img)
	x,y,w,h = cv2.boundingRect(maxContour)

	#cv2.rectangle(clone, (x,y), (x+w, y+h), 2)
	cv2.rectangle(cropped_img, (x,y), (x+w, y+h), 2)

	#drawing the contours
	hull = cv2.convexHull(maxContour)


	drawing = np.zeros(cropped_img.shape, np.uint8)
	cv2.drawContours(drawing, [maxContour], -1, (0,255,0), 0)
	#cv2.fillPoly(drawing, pts =[maxContour], color=(255,255,255))
	thresholded = thresholded/255.0
	gray = img/255.0
	


	#print(predicted)

	cv2.putText(clone,
                   '%s ' % (labels[predicted]),
                   (10, 60), cv2.FONT_HERSHEY_PLAIN,3,(0, 0, 0))
	#cv2.imshow("Gray", img)
	cv2.imshow("Result", img)
	cv2.imshow("Original", clone)

	if(cv2.waitKey(25) & 0xff == ord('q')):
		break

cap.release()
cv2.destroyAllWindows()