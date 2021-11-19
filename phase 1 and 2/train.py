import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2
import os
import numpy as np
from glob import glob
from tensorflow.keras.callbacks import ModelCheckpoint


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.mobilenet import preprocess_input
from IPython.display import Image


def get_image_size():
	img = cv2.imread('gestures/1/100.jpg', 0)
	return img.shape

def get_num_images(num):
	return len(glob('gestures/{}/*'.format(num)))

image_x, image_y = get_image_size()

def get_num_of_classes():
	return len(glob('gestures/*'))

def create_model():
	num_of_classes = get_num_of_classes()
	model = Sequential()
	model.add(Conv2D(16, (2,2), input_shape=(image_x, image_y, 1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
	model.add(Conv2D(32, (3,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
	model.add(Conv2D(64, (5,5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(num_of_classes, activation='softmax'))
	checkpoint1 = ModelCheckpoint("modelPleaseWork.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint1]
	# from tensorflow.keras.utils import plot_model
	# plot_model(model, to_file='model.png', show_shapes=True)
	return model, callbacks_list

def create_dataset():
	train_data_array = []
	train_labels = []
	val_data_array = []
	val_label_array = []
	for imageFolders in os.listdir('gestures'):
		if(imageFolders == ".DS_Store"):
			continue
		print(imageFolders)
		num_images = get_num_images(imageFolders)
		num_val = int(0.1*num_images)
		print("Reading images for {}".format(imageFolders))
		count = 0
		for imageFile in os.listdir(os.path.join('gestures', imageFolders)):
			image_path = os.path.join('gestures', imageFolders, imageFile)
			img = cv2.imread(image_path, 0)
			img = np.array(img)
			img = img/255.0
			img = img.astype('float32')
			if count==num_val:
				val_data_array.append(img)
				val_label_array.append(int(imageFolders))
			
			train_data_array.append(img)
			train_labels.append(int(imageFolders))
			count+=1
	return train_data_array, train_labels, val_data_array, val_label_array

def train():
	X_data, Y_data, val_x, val_y = create_dataset()
	X_data = np.reshape(X_data, (len(X_data), image_x, image_y,1))
	model, callbacks_list = create_model()
	model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=0.01), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
	history = model.fit(x=np.array(X_data, np.float32), y=np.array(list(map(int,Y_data))), verbose=1, epochs=20, batch_size=500, callbacks=callbacks_list)

	model.save('modelTest2.h5')

def make_prediction(modelname):
	model = tf.keras.models.load_model(modelname)
	X_data, Y_data, val_x, val_y = create_dataset()
	X_data = X_data[10505]
	X_data = np.reshape(X_data, (1,50,50))
	# X_data = np.reshape(X_data, (1,50,50,1))
	X_data = np.repeat(X_data[..., np.newaxis], 3, -1)
	print("Predicted: ", np.argmax(model.predict(X_data)))
	print("Actual: ", Y_data[10505])



def transfer_learning_training():
	num_of_classes = get_num_of_classes()
	vgg_base = VGG16(weights='imagenet',include_top=False,input_shape=(image_x,image_y,3))
	for layer in vgg_base.layers:

		layer.trainable = False
	model = Sequential()
	model.add(vgg_base)
	model.add(Flatten())
	model.add(Dense(512,activation='relu'))
	model.add(Dropout(0.8))
	model.add(Dense(256,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_of_classes, activation='softmax'))

	X_data, Y_data, val_x, val_y = create_dataset()
	X_data = np.reshape(X_data, (len(X_data), image_x, image_y))
	x = np.repeat(X_data[..., np.newaxis], 3, -1)
	x = np.array(x, np.float32)
	y=np.array(list(map(int,Y_data)))
	val_x = np.reshape(val_x, (len(val_x), image_x, image_y))
	val_x = np.repeat(val_x[..., np.newaxis], 3, -1)
	val_x = np.array(val_x, np.float32)
	val_y = np.array(list(map(int,val_y)))



	model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),    metrics=['accuracy'])
	checkpoint = ModelCheckpoint("Weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
	history = model.fit(x,y, batch_size=32, epochs=10, verbose=1,validation_data=(val_x,val_y), shuffle=True,callbacks=[checkpoint])

	model.save("Model/model.h5")

	model_json = model.to_json()
	with open("Model/model.json", "w") as json_file:    
		json_file.write(model_json)

	model.save_weights("Model/model_weights.h5")


make_prediction('Model/model.h5')