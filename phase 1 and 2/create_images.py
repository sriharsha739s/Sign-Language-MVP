import cv2
import os
import time
import keyboard

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


def create_images(num_images):
	labels = [0, 1, 2, 3, 4, 5]
	letters = ['A', 'B', 'C', 'D','E',' F']
	for label in labels:
		if not os.path.exists("images/{}".format(label)):
			os.makedirs("images/{}".format(label))
	k = 0
	for imageFolders in os.listdir('images'):
		if(imageFolders == ".DS_Store"):
			continue
		cap = cv2.VideoCapture(0)
		print("Collecting images for {}".format(letters[k]))
		time.sleep(5)
		
		for i in range(num_images):
			print('Collecting image {}'.format(i))
			success,frame = cap.read()
			frame = cv2.flip(frame, 1)

			cv2.rectangle(frame, (500,500), (50,50), (0,255,0),0)
			cv2.rectangle(frame, (500,500), (50,50), (0,255,0),0)
			cropped_img = frame[50:500, 50:500]

			img = preprocess(cropped_img)

			imgname = os.path.join("images", str(imageFolders), str(i) + '.jpg')
			
			cv2.imshow("Gray", img)
			cv2.imshow("Original", frame)
			time.sleep(1)
			cv2.imwrite(imgname, img)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		k = k+1

		cap.release()
		cv2.destroyAllWindows()



create_images(5)
