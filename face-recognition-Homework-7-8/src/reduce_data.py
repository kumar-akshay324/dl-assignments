# import shut	
import os
from numpy.random import randint
import random
from shutil import copyfile, copy2

def create_data(train_size, test_size, valid_size, path_to_images):

	if (train_size !=0):
		for item in os.listdir(path_to_images):

			class_name = path_to_images + "/" + item

			images_in_class = os.listdir(class_name)
			num_images = len(images_in_class)

			train_folder_name = create_new_dirs(0, item)

			if (train_folder_name == ""):
				print ("Task Failed!")

			random_list = random.sample(range(0, num_images), train_size)
			# print ("Random List: %s" %(str(random_list)))

			for index, image_file in enumerate(images_in_class):
				
				image_filename = class_name + "/" + image_file

				if (index in random_list):
					copy2(image_filename, train_folder_name)

		print ("Created Train Dataset")

	if (test_size !=0):
		for item in os.listdir(path_to_images):

			class_name = path_to_images + "/" + item

			images_in_class = os.listdir(class_name)
			num_images = len(images_in_class)

			test_folder_name = create_new_dirs(1, item)

			if (test_folder_name == ""):
				print ("Task Failed!")

			random_list = random.sample(range(0, num_images), int(test_size))
			# print ("Random List: %s" %(str(random_list)))

			for index, image_file in enumerate(images_in_class):
				
				image_filename = class_name + "/" + image_file

				if (index in random_list):
					copy2(image_filename, test_folder_name)

		print ("Created Test Dataset")

	if (valid_size !=0):
		for item in os.listdir(path_to_images):

			class_name = path_to_images + "/" + item

			images_in_class = os.listdir(class_name)
			num_images = len(images_in_class)

			valid_folder_name = create_new_dirs(2, item)

			if (valid_folder_name == ""):
				print ("Task Failed!")

			random_list = random.sample(range(0, num_images), valid_size)
			# print ("Random List: %s" %(str(random_list)))

			for index, image_file in enumerate(images_in_class):
				
				image_filename = class_name + "/" + image_file

				if (index in random_list):
					copy2(image_filename, valid_folder_name)

		print ("Created Valid Dataset")

def create_new_dirs(data_type_index, class_foldername):

	data_type = ["train", "test", "valid"]
	if (data_type_index == 0):
		folder_name = os.getcwd() + "/data/" + data_type[data_type_index] + "/" + class_foldername
	elif (data_type_index == 1):
		folder_name = os.getcwd() + "/data/" + data_type[data_type_index] + "/" + class_foldername
	elif (data_type_index == 2):
		folder_name = os.getcwd() + "/data/" + data_type[data_type_index] + "/" + class_foldername

	# Setup Output Folder
	if not os.path.isdir(folder_name):
		# print ("Path doesn't exist")
		try:
			os.makedirs(folder_name)
			print ('\033[94m' + "Made a new folder named %s" %folder_name)
		except OSError as exc: # Guard against race condition
			pass
	else:
		print ('\033[93m' + "Image Output Folder already exists")

	return folder_name

if __name__ == '__main__':
	path_to_images =  os.getcwd() +  "/images"
	create_data(0, 50, 0, path_to_images)




