
from keras.models import Sequential
from keras.layers.core import Dense
from keras.datasets import mnist

from keras.layers.convolutional import *
from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import *

from keras import backend as K

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt 

class RecogniseHandWrittenDigits():

	def __init__(self):
		jb = 4

	def loadData():
		(self.x_train, self.y_train), (self.x_test, y_test) =  mnist.load_data()

	def create_model():
		model = Sequential()
		# model.add(Dropout(0.2), input_dim = )





