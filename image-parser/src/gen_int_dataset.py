import os
import xml.etree.ElementTree as elemTree

def parse_xml(file_to_parse):
	
	print (os.getcwd())
	xml_tree = elemTree.parse(file_to_parse)
	xml_tree_root = xml_tree.getroot()
	print ("I was here")
	print (xml_tree_root.tag)

def parse_files(directory_name):

	for file_n in os.listdir(directory_name):
		if file_n.endswith(".xml"):
			parse_xml( directory_name + "/" + file_n)
		break
def 



dir_name = "../VOCdevkit/VOC2012/Annotations"
parse_files(dir_name)