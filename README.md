# Image-Segmentation-using-Mask-RCNN-and-OpenCV
Applying a segmentation model to celebrity face dataset using Mask-RCNN and openCV.

# **Image Segmentation using Mask-RCNN and OpenCV** 

---

**Image Segmentation Challenge**

The steps of this project are the following:
* Construct the argument parse and parse the arguments
* load the COCO class labels the Mask R-CNN was trained on
* load the Mask R-CNN trained on the COCO dataset
* load the input image and make a prediction on it
* Visualize the extracted ROI, the mask, along with the segmented instance

The project directories are the following:
* mask-rcnn-coco/: The Mask R-CNN model files. There are three files:
  * frozen_inference_graph.pb: The Mask R-CNN model weights. The weights are pre-trained on the COCO dataset.
  * mask_rcnn_inception_v2_coco_2018_01_28.pbtxt: The Mask R-CNN model configuration.
  * object_detection_classes_coco.txt: All 90 classes are listed in this text file, one per line. 
* images/: contains the zip file of the CelebA dataset
*mask_rcnn.py: This is the main script and it will perform instance segmentation and apply a mask to the image so you can see where the Mask R-CNN thinks an object is.

[//]: # (Image References)

[image1]: ./relevant_images/training_distribution.png "Training Visualization"
[image2]: ./relevant_images/validation_distribution.png "Validation Visualization"
[image3]: ./relevant_images/testing_distribution.png "Testing Visualization"
[image4]: ./test_images/14.jpg "Traffic Sign 5"
[image5]: ./test_images/11.jpg "Traffic Sign 4"
[image6]: ./test_images/12.jpg "Traffic Sign 1"
[image7]: ./test_images/15.jpg "Traffic Sign 2"
[image8]: ./test_images/18.jpg "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410 
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data labels are distributed in the entire data

![alt text][image1] ![alt text][image2] ![alt text][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.


I normalized the image data to speeds up learning and achieve faster convergence.

All images were normalized to values between [-1 1]

