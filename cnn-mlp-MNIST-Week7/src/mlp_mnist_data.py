#!usr/bin/env

import numpy as np
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

from keras import backend as K
K.set_image_dim_ordering('th')

class RecogniseHandWrittenDigits():

	def __init__(self, epochs, b_size, vbose):
		self.num_classes = None
		self.num_epochs = epochs
		self.size_batches = b_size
		self.verbosity = vbose

	def loadData(self):
		(self.x_train, self.y_train), (self.x_test, self.y_test) =  mnist.load_data()

	def prepareData(self):
		self.num_pixels = self.x_train.shape[1] * self.x_train.shape[2]
		self.x_train = self.x_train.reshape(self.x_train.shape[0], self.num_pixels).astype('float32')
		self.x_test = self.x_test.reshape(self.x_test.shape[0], self.num_pixels).astype('float32')
		self.x_train, self.x_test = self.x_train/255.0, self.x_test/255.0

	def prepareLabels(self):
		self.y_train = np_utils.to_categorical(self.y_train)
		self.y_test = np_utils.to_categorical(self.y_test)
		self.num_classes = self.y_test.shape[1]

	def createOptimizer(self, opt):

		if opt=="adam":
			designed_optimizer = Adam(lr=0.0001) #, beta_1=0.9999, beta_2=0.999)
			# designed_optimizer = Adam()
		elif opt=="sgd":
			designed_optimizer = SGD(lr=0.1)

		return designed_optimizer

	def createModelMLP(self):
		model = Sequential()

		model.add(Dense(self.num_pixels, input_dim=self.num_pixels, init='normal', activation='relu'))

		model.add(Dense(256, activation="relu"))
		model.add(Dropout(0.2))

		model.add(Dense(self.num_classes, activation="softmax"))

		adm = self.createOptimizer("adam")
		model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=adm)
		print (model.summary())

		return model

	def trainModel(self, model):

		model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=self.num_epochs, batch_size=self.size_batches)
		self.trained_model = model

	def testModel(self):
		self.training_score = self.trained_model.evaluate(self.x_test, self.y_test, verbose=self.verbosity)

	def printScore(self):

		print("MLP Error: %.2f%%" % (100-self.training_score[1]*100))

if __name__ == '__main__':

	epochs, b_size, vbose = 50, 200, 2 
	mnist_obj = RecogniseHandWrittenDigits(epochs, b_size, vbose)
	mnist_obj.loadData()
	mnist_obj.prepareData()
	mnist_obj.prepareLabels()
	created_model = mnist_obj.createModelMLP()
	mnist_obj.trainModel(created_model)
	mnist_obj.testModel()
	mnist_obj.printScore()



