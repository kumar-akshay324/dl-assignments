import cv2
import numpy as np
import os, math, time
import pandas as pd
import matplotlib.pyplot as plt


class ExtractFaceFromImage():

	def __init__(self, path_to_image_folder):
		self.image_folder_path = path_to_image_folder

	def cascaded_filters(self):
		cascade_file_src = "haarcascade_frontalface_default.xml"
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
			image = cv2.imread(image_file, cv2.LOAD_IMAGE_COLOR)
			# # image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			print (type(image))

			# cv2.imshow("ImageWindow", image)
			# cv2.waitKey()

			# # Find all available faces in the image
			# detected_faces = faceCascade.detectMultiScale(gray, 1.2, 5)

			# # Bounding Box around the face
			# for (l_x, l_y, l_w, l_h) in detected_faces:
			#     cv2.rectangle(image, (l_x, l_y), (l_x+l_w, l_y+l_h), (0, 255, 0), 2)

			# plt.imshow(image)

	def execute(self):
		self.cascaded_filters()
		self.cascaded_classifier()

if __name__ == '__main__':
	current_path  = os.getcwd()
	image_path  = current_path + "/" + "images/akshay1"
	faceObj = ExtractFaceFromImage(image_path)
	faceObj.execute()