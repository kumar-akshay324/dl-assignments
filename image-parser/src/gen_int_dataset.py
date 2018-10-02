import os
import xml.etree.ElementTree as elemTree
from PIL import Image 

class AnnotatedImageParser():

	def __init__(self, annotation_dir, image_dir):
		self.directory_name = annotation_dir
		self.image_dir_name = image_dir

	def parse_xml(self, file_to_parse):
		
		print (os.getcwd())
		xml_tree = elemTree.parse(file_to_parse)
		xml_tree_root = xml_tree.getroot()
		return xml_tree_root

	def parse_files(self):

		for file_n in os.listdir(self.directory_name):
			if file_n.endswith(".xml"):
				self.file_n = file_n
				xml_root = self.parse_xml( self.directory_name + "/" + self.file_n)
				self.print_all_tags(xml_root)
			break

	def print_all_tags(self, root):

		print ("Root Tag: ", root.tag)

		for prt in root.iter("part"):
			name = prt.find("name")
			bound_box =  prt.find("bndbox")

			bound_box_coordinates = {"xmin": 0, "xmax": 0, "ymin": 0, "ymax": 0}

			for k in bound_box_coordinates.keys():
				bound_box_coordinates[k] = float(bound_box.findtext(k))

			print ("SubImage Name: %s  | Bounding box coordinates: %s" %(name.text, str(bound_box_coordinates)))

			extracted_image = self.extractSubimage(bound_box_coordinates)
			self.putSubImageInFolder(name.text, extracted_image)

	def extractSubimage(self, bound_box_coordinates):
		pass
		image_file_name = self.image_dir_name + "/" + self.file_n.split(".")[0] + ".jpg"
		image_file  = Image.open(image_file_name)
		print ("Image File Name: %s" %image_file_name)

		img_bb = (bound_box_coordinates["xmin"], bound_box_coordinates["ymax"], bound_box_coordinates["xmax"], bound_box_coordinates["ymin"])
		print ("---", img_bb)

		cropped_image = image_file.crop(img_bb)
		cropped_image.show()

		return True



	def putSubImageInFolder(self, a, b):
		pass


if __name__ == '__main__':
	
	a_dir_name = "../VOCdevkit/VOC2012/Annotations"
	i_dir_name = "../VOCdevkit/VOC2012/JPEGImages"
	annotation_parser = AnnotatedImageParser(a_dir_name, i_dir_name)
	annotation_parser.parse_files()
