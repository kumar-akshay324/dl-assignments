Assignments for course Special Topics - RBE 595: Deep Learning For Advanced Robot Perception at Worcester Polytechnic Institute

# Support Vector Machines - Week 1

Implementing a Support Vector Machine based linear image classifier for the CIFAR-10 dataset using Stochastic Gradient Descent optimization. The dataset has 50000 images for 10 classes of images (car, cat, deer, frog, ...)

### Usage

* Download the dataset by running `./getdataset.sh` shell script in _SVM-Week1/f16RBE595/data_ 
* Run the `SVM.py` python script to obtain results.

PS - The shell script might not run unless the privileges are correct and it is declared as an executable(Linux)

### Results

As expected, due to poor resolution of the images(32 X 32 X 3) and this being a very naive implementation of the algorithm, the results obtained were not very good.

* Training Accuracy: ~0.38
* Testing Accuraccy: ~0.41

The final template generated by the weight matrix for all the classes shown below indicates a few classes like car, horse, truck and frog resemble a general representation satisfactorily while other class templates are  not very clear. 

![alt text](https://github.com/kumar-akshay324/dl-assignments/raw/master/SVM-Week1/results/SVM_results_CIFAR-10.png "Final Template Generated from weights")

![alt text](https://github.com/kumar-akshay324/dl-assignments/raw/master/SVM-Week1/results/loss_iters.png "Variation in Loss over the several iterations")


# Image Parser - Week 6

Generate an intermediate dataset by parsing the XML annotation files for given VOC PASCAL 2012 image dataset. The parser extracts subimages from the JPEGImages folder using the annotations provided in the dataset. [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit)

### Usage

* Launch the gen_int_dataset.py python file from within the src/ folder. Uncomment the line #114 to run the complete parser 
* The VOC Dataset should be downloaded and stored one folder above the python script, i.e. in the same folder as the src/ folder. 
* The final generated dataset will be stored in this same folder with the name classes/. The classes folder will have another 20 individual folders representing the 20 classes and store the corresponding images. 

* A binary will also be generated for the images using the _Pickle_ library  in Python. Storing the image information in a serialized manner shall help make its usage in ML applications easier. Generated binary should have the name p_dataset.pkl

## File Structure

	- Image-parser-Week6
		- src
		- classes
			- aeroplane
			..
			..
			- cat 
		- VOCdevkit

### Results

* All the images parsed to subimages and stored within individual class folder have been resized to 224 x 224 pixel resolution with restored quality


# Multi-Layer Perceptron and Convolutional Neural Networks for MNIST Handwritten Digit recognition

Create an basic MLP neural network and another CNN to classify handwritten digits from the MNIST data available by default through the Keras library. 

### TO BE UPDATED!

### Usage

## Results


