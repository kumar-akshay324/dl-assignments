#!usr/bin/env

import numpy as np
import os
from keras.models import Sequential
from keras.layers.core import Dense
from keras.datasets import mnist

from keras.layers.convolutional import *
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import *

from keras import backend as K

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

from keras.utils import np_utils
from keras.optimizers import *
import pandas as pd

from keras import backend as K
K.set_image_dim_ordering('th')

#from VideoToImages import *

class FaceRecognition():

	# extract_images = VideoToImages()

	def __init__(self, epochs, b_size, vbose, folder_paths):
		self.num_epochs = epochs
		self.size_batches = b_size
		self.verbosity = vbose
		self.train_path, self.test_path, self.valid_path = folder_paths

	def loadData(self):
		train_datagen = ImageDataGenerator() #rescale=1./255, shear_range=0.2, zoom_range=0.2)
		test_datagen = ImageDataGenerator() #rescale=1./255)

		print ("Train Path: %s" %(self.train_path))
		print ("Test Path: %s" %(self.test_path))
		
		self.train_batches = train_datagen.flow_from_directory(directory=self.train_path,
																target_size=(148, 148),
																color_mode="rgb",
																# class_mode="binary",
																batch_size=self.size_batches)
																
		self.valid_batches = test_datagen.flow_from_directory(directory=self.valid_path,
																target_size=(148, 148),
																color_mode="rgb",
																# class_mode="categorical",
																batch_size=self.size_batches)
		# self.valid_batches = self.train_batches

		self.test_batches = test_datagen.flow_from_directory(directory=self.test_path,
																target_size=(148, 148),
																color_mode="rgb",
																class_mode=None,
																batch_size=self.size_batches)

	def prepareData(self):

		print ("Type of Training Batch Data: %s" %(self.train_batches))

		# self.train_batches = len(self.train_batches)
		sizee = len(self.train_batches)

		print ("Length of batches: ", sizee)

	def createOptimizer(self, opt):

		lr = 0.001
		dcy = lr/self.num_epochs
		if opt=="adam":
			designed_optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=dcy)
			#designed_optimizer = Adam()
		elif opt=="sgd":
			designed_optimizer = SGD(lr=0.1)

		return designed_optimizer

	def createModel(self):
		model = Sequential()
		model.add(Conv2D(32,(3,3), activation="relu", input_shape=(3,148, 148), padding="valid"))
		model.add(MaxPooling2D(pool_size=(3,3)))
		model.add(Conv2D(64,(3,3), activation="sigmoid", padding="valid"))
		model.add(MaxPooling2D(pool_size=(3,3)))
		model.add(Conv2D(128,(3,3), activation="relu", padding="valid"))
		model.add(MaxPooling2D(pool_size=(3,3)))
		# model.add(Conv2D(256,(3,3), activation="sigmoid", padding="valid"))
		# model.add(MaxPooling2D(pool_size=(2,2)))
		# model.add(Conv2D(128,(3,3), activation="relu", padding="valid"))
		# model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Flatten())
		# model.add(Dense(64, activation="sigmoid"))
		model.add(Dense(32, activation="relu"))
		model.add(Dense(len(self.train_batches.class_indices), activation="softmax"))

		adam_optimizer = self.createOptimizer("adam")
		model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=adam_optimizer)

		# Print the model architecture
		print (model.summary())
		return model

	def trainModel(self, model):

		TRAIN_STEP_SIZE = self.train_batches.n//self.train_batches.batch_size
		VALID_STEP_SIZE = self.valid_batches.n//self.valid_batches.batch_size
		
		model.fit_generator(generator=self.train_batches,
							steps_per_epoch=TRAIN_STEP_SIZE,
							validation_data=self.valid_batches,
							validation_steps=VALID_STEP_SIZE,
							epochs=self.num_epochs,verbose=self.verbosity)

		self.trained_model = model
		self.trained_model.save("cnn_face_recognition.h5")

	def evaluateModel(self):
		self.trained_model.evaluate_generator(generator=self.valid_batches)

	def testModel(self):
		# Reset the test batches to get the results in correct order
		self.test_batches.reset()
		# Obtain predictions for the images
		preds = self.trained_model.predict_generator(self.test_batches, verbose=self.verbosity)
		# print ("test Data---", len(self.train_batches.class_indices))
		print ("---", preds)
		# print ("---", len(self.train_batches))
		# Obtain indices from prediction because these would be categorical or one hot encoding and 
		# need to be changed to class names
		predicted_class_indices = np.argmax(preds,axis=1)

		labels = (self.train_batches.class_indices)
		labels = dict((v,k) for k,v in labels.items())

		# # Final folder name based predictions
		self.predictions = [labels[k] for k in predicted_class_indices]

	def savePredictions(self):
		filenames = self.test_batches.filenames
		results = pd.DataFrame({"Filename":filenames,
							  "Predictions":self.predictions})
		results.to_csv("results.csv",index=False)

	def execute(self):

		self.loadData()
		self.prepareData()
		created_model = self.createModel()

		if not os.path.isfile(os.getcwd() + ("/cnn_face_recognition.h5")):
			print ("Training the model from scratch")
			self.trainModel(created_model)
		else:
			print ("Using a trained model weights")
			self.trained_model = created_model
			self.trained_model.load_weights("cnn_face_recognition.h5")

		self.evaluateModel()
		self.testModel()
		self.savePredictions()

if __name__ == '__main__':
	
	train_path = os.getcwd() + "/data/train"
	test_path = os.getcwd() + "/data/test"
	valid_path = os.getcwd() + "/data/valid"
	
	paths = [train_path, test_path, valid_path]
	
	epochs, b_size, vbose = 1, 5, 1
	face_recognition = FaceRecognition(epochs, b_size, vbose, paths)
	face_recognition.execute()



