import cv2
import numpy as np
import os, math, time
import pandas as pd
import matplotlib.pyplot as plt
# from PIL import Image as imm

class ExtractFaceFromImage():

	def __init__(self, path_to_image_folder):
		self.image_folder_path = path_to_image_folder

	def cascaded_filters(self):
		cascade_file_src = "/home/akshay/anaconda3/envs/deeplearning/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml"
		# cascade_file_src = "haarcascade_frontalface_default.xml"
		self.faceCascade = cv2.CascadeClassifier(cascade_file_src)

	def cascaded_classifier(self):

		print (self.image_folder_path)
		for image_file in os.listdir(self.image_folder_path):

			image_filename = self.image_folder_path + "/" + image_file

			if not os.path.isfile(image_filename):
				print ("Image File does not exist")
			else:
				print ("Image File Name: ", image_filename)

			# # Load image and convert into grayscale
			image_grayscale = cv2.imread(image_filename,0) #cv2.WINDOW_NORMAL)
			# image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

			# cv2.imshow("", image_grayscale)
			# cv2.waitKey()

			# # Find all available faces in the image
			detected_faces = self.faceCascade.detectMultiScale(image_grayscale, 1.2, 5)

			# # Bounding Box around the face
			for (l_x, l_y, l_w, l_h) in detected_faces:
			    cv2.rectangle(image_grayscale, (l_x, l_y), (l_x+l_w, l_y+l_h), (125, 255, 50), 2)

			image_grayscale = cv2.resize(image_grayscale, (image_grayscale.shape[0]//5, image_grayscale.shape[1]//5))
			cv2.imshow("ImageWindow",image_grayscale)
			cv2.waitKey()

	def execute(self):
		self.cascaded_filters()
		self.cascaded_classifier()

if __name__ == '__main__':
	current_path  = os.getcwd()
	image_path  = current_path + "/" + "images/akshay1"
	faceObj = ExtractFaceFromImage(image_path)
	faceObj.execute()