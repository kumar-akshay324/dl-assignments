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
			
			
	# returns "True" if face is detected in image stored at img_path
	def face_detector(img_path):
    		img = cv2.imread(img_path)
    		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    		faces = face_cascade.detectMultiScale(gray)
    		return len(faces) > 0
	
###################	
from PIL import ImageFile 
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint 
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255



model = Sequential()

### TODO: Defining architecture.

model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
 

### TODO: specify the number of epochs to train the model.

epochs = ...

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)
model.load_weights('saved_models/weights.best.from_scratch.hdf5')

# get index of predicted person for each image in test set
person_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(person_predictions)==np.argmax(test_targets, axis=1))/len(person_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
####################################
if __name__ == '__main__':
	video = os.getcwd() + "/videos/akshay.mp4"
	frames_per_second = 30
	images_folder = "../extracted_images"
	vidImgObj = VideoToImages(video, frames_per_second)
	vidImgObj.processFolder()
	vidImgObj.extractImagesFromVideo()
