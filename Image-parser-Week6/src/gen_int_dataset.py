import os
import xml.etree.ElementTree as elemTree
from PIL import Image 
import time
import pickle
class AnnotatedImageParser():

	def __init__(self, annotation_dir, image_dir):
		self.directory_name = annotation_dir
		self.image_dir_name = image_dir
		self.class_folder_name = "../classes"
		self.createFolder(self.class_folder_name)

	def parse_xml(self, file_to_parse):
		
		# print ("Current Working Directory: %s " %os.getcwd())
		xml_tree = elemTree.parse(file_to_parse)
		xml_tree_root = xml_tree.getroot()
		return xml_tree_root

	def parse_files(self):

		count = 0
		for file_n in os.listdir(self.directory_name):
			count += 1
			if file_n.endswith(".xml"):
				self.file_n = file_n
				xml_root = self.parse_xml( self.directory_name + "/" + self.file_n)
				self.print_all_tags(xml_root)
			# if (count >5):
			# 	break

	def print_all_tags(self, root):

		# print ("Root Tag: ", root.tag)

		for prt in root.iter("object"):
			name = prt.find("name")
			bound_box =  prt.find("bndbox")

			bound_box_coordinates = {"xmin": 0, "xmax": 0, "ymin": 0, "ymax": 0}

			for k in bound_box_coordinates.keys():
				bound_box_coordinates[k] = float(bound_box.findtext(k))

			# print ("SubImage Name: %s  | Bounding box coordinates: %s" %(name.text, str(bound_box_coordinates)))

			extracted_image = self.extractSubimage(bound_box_coordinates)
			i =5
			self.putSubImageInFolder(name.text, extracted_image)

	def extractSubimage(self, bound_box_coordinates):
		image_file_name = self.image_dir_name + "/" + self.file_n.split(".")[0] + ".jpg"
		# print ("Image File Name: %s" %image_file_name)

		original_file  = Image.open(image_file_name)
		image_file = original_file.copy()
		img_bb = (bound_box_coordinates["xmin"], bound_box_coordinates["ymin"], bound_box_coordinates["xmax"], bound_box_coordinates["ymax"])

		cropped_image = image_file.crop(img_bb)
		cropped_image = cropped_image.resize((224, 224), Image.ANTIALIAS)
		return cropped_image

	def createFolder(self, f_name):

		folder_name = os.getcwd() + "/" + 	f_name
		# print ("-----", os.getcwd())
		# print ("===========", folder_name)
		# print (os.path.isdir(folder_name))

		# Setup Output Folder
		if not os.path.isdir(folder_name):
			# print ("Path doesn't exist")
			try:
				os.makedirs(folder_name)
				# print ("Made a new folder named %s" %folder_name)
			except OSError as exc: # Guard against race condition
				pass
				print ("Didn't create a new folder")
		return True

	def putSubImageInFolder(self, subimage_folder_name, b):

		final_folder = self.class_folder_name + "/" + subimage_folder_name
		# print ("Final Folder: %s" %final_folder)
		if self.createFolder(final_folder):
			num_image = len(os.listdir(final_folder)) + 1;
			final_image_name = final_folder + "/" + str(num_image) + ".jpg"
			b.save(final_image_name, "JPEG", quality=100)
			return True
		else:
			print ("False")
			return False

	def binary_creation(self):

		with open('parsed_datset.pkl','ab') as p_dataset:
		    for s_class in os.listdir('../classes/'):
		    	subclass = os.getcwd() + '/../classes/' + str(s_class)
		    	print (subclass)

		    	for subclass_img in os.listdir(subclass):
		            image_filename = str(subclass) + "/" + str(subclass_img) 
		            img = Image.open(image_filename,'r')
		            pickle.dump(img, p_dataset)
		            img.close()
		    p_dataset.close()

if __name__ == '__main__':
	
	a_dir_name = "../VOCdevkit/VOC2012/Annotations"
	i_dir_name = "../VOCdevkit/VOC2012/JPEGImages"
	annotation_parser = AnnotatedImageParser(a_dir_name, i_dir_name)
	# annotation_parser.parse_files()
	annotation_parser.binary_creation()
	print ("Parsed all Images")
