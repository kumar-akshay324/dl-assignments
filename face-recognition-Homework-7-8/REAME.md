# Face Recognition Using DeepLearning Python

THIS IS STILL UNDER HEAVY DEVELOPMENT!

A complete face recognition pipeline that takes in various videos of people, parses them to images as per input frames per second available in the video, extracts the face from the complete image using OpenCV cascade filters and then uses DeepLearning (Convolutional Neural Networks) to classify the faces for different people.

## File Structure

```
├── face-recognition-Homework-7-8
	├── cnn_face_recognition.h5
	├── Makefile
	├── REAME.md
	├── data
	│   ├── train
	│   │   ├── classA
	│   │	│	├── imagea.png
	│   │   │	└── imageb.png
	│   │   └── classB
	│   │		├── imagec.png
	│   │   	└── imaged.png	
	│   ├── test
	│   │   ├── image1.png
	│   │   └── image2.png
	│   └── valid
	│   	├── image3.png
	│       └── image4.png
	├── images
	│   ├── classA
	│   ├── classB
	│   └── classC
	├── src
	│   ├── extract_faces.py
	│   ├── extract_video_frames.py
	│   └── face_recognition_cnn.py
	└── videos
	    ├── classA
	    │   ├── video1.mp4
	    │   ├── video2.mp4
	    │   ├── video3.mp4
	    │   └── video4.mp4
	    ├── classB
	    │   ├── video1.mp4
	    └── classC
	        ├── video3.mp4
	        └── video4.mp4
```
## File Description

* _extract_video_frames.py_: Extracts individual frames from videos and creates dataset 
* _extract_faces.py_: Extracts only the face region from any image
* _face_recognition_cnn.py_: Uses the CNN deep network to run classification 
* _reduce_data.py_: Extracts certain number images from the complete dataset and creates Keras processible dataset with training, testing and validation data

## Usage

The complete pipeline is based on _make_ utility system and uses a _Makefile_

All the `make` commands should be run from the base directory, i.e. from _face-recognition-Homework-7-8_

### Data Preparation

* Record videos of the face/face-focused body and store them inside the _videos_ folder within the sub-folder _class_name_
* Run command `make install-dependencies` to install all the system dependencies
* Run command `make images` to generate dataset with images extracted from all the videos and accumulated below _images/class_name_
* Optional -  Run command `make detect-faces` to extract the specific facial zones from all the images generated above to working on specifically facial regions  in the images as a simplified image classification problem

* Run command `make reduce-dataset` to create a version of the as proper data under train, test and valid folders

### Face Recognition

* Run command `make recognize_faces` to run a CNN deep network that works on the supplied images to classify images on the classes from the training data.
