import cv2
import numpy
import math, os, time

class VideoToImages():

	def __init__(self, video, frames_per_second):
		self.video_filename = video
		self.desired_fps = frames_per_second

	def processFolder(self):

		# print ("Video File Name: ", self.video_filename)
		if os.path.isfile(self.video_filename):
			print ('\033[92m' + "Video File Found")
		else:
			print ('\033[93m' + "Video file does not exist")

		image_folder_name = (((self.video_filename.split("/"))[-1]).split("."))[0]
		folder_name = os.getcwd() + "/images/" + image_folder_name
		# print ('\033[94m' + "Image folder name: ", folder_name)

		self.images_folder = folder_name
		# Setup Output Folder
		if not os.path.isdir(folder_name):
			# print ("Path doesn't exist")
			try:
				os.makedirs(folder_name)
				print ('\033[94m' + "Made a new folder named %s" %folder_name)
			except OSError as exc: # Guard against race condition
				pass
		else:
			print ('\033[92m' + "Image Output Folder already exists")

	def extractImagesFromVideo(self):
		'''https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html'''
		capture = cv2.VideoCapture(self.video_filename)
		frame_rate = capture.get(cv2.CAP_PROP_FPS)
		while (capture.isOpened()):
			return_flag, image_frame = capture.read()
			frame_id = capture.get(cv2.CAP_PROP_POS_FRAMES)
			if (return_flag):

				# print ("Image Type: ", type(image_frame))
				# print ("Frame Rate: ", frame_rate)
				# print ("Image Read Status: ", "Success" if return_flag else "Failed")
				# cv2.imshow("ImageWindow", image_frame)
				# cv2.waitKey()

				success_list = []
				# if (frame_id % math.floor(frame_rate) == 0):
				img_filename = self.images_folder + "/image_" + str(int(frame_id)) + ".png"
				if os.path.isfile(img_filename):
					continue
				ret_value = cv2.imwrite(img_filename, image_frame)
				success_list.append(ret_value)
			else:
				break
	
		if all(success_list):
			print ('\033[92m' + "Success storing images")
		else:
			print ('\033[93m' + "Could not save all images")

if __name__ == '__main__':
	video = os.getcwd() + "/videos/akshay.mp4"
	frames_per_second = 30
	images_folder = "../extracted_images"
	vidImgObj = VideoToImages(video, frames_per_second)
	vidImgObj.processFolder()
	vidImgObj.extractImagesFromVideo()