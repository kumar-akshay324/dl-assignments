import cv2
import numpy
import math, os

class VideoToImages():

	def __init__(self, video, frames_per_second):
		self.video_filename = video
		self.fps = frames_per_second

	def processFolder(self):

		print ("Video File Name: ", self.video_filename)
		if not os.path.isdir(self.video_filename):
			print ("Video file does not exist")

		image_folder_name = (((self.video_filename.split("/"))[-1]).split("."))[0]
		folder_name = os.getcwd() + "/../images/" + image_folder_name
		print ("Image folder name: ", folder_name)

		self.images_folder  = folder_name

		# Setup Output Folder
		if not os.path.isdir(folder_name):
			# print ("Path doesn't exist")
			try:
				os.makedirs(folder_name)
				# print ("Made a new folder named %s" %folder_name)
			except OSError as exc: # Guard against race condition
				pass

	def processVideo(self):
		capture = cv2.VideoCapture(video_filename)
		return_flag, image_frame = capture.read()
		frame_rate = capture.get()
		print ("Frame Rate: ", frame_rate)


if __name__ == '__main__':
	video = os.getcwd() + "/../videos/akshay.mp4"
	frames_per_second = 30
	images_folder = "../extracted_images"
	vidImgObj = VideoToImages(video, frames_per_second)
	vidImgObj.processFolder()