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
* mask_rcnn.py: This is the main script and it will perform instance segmentation and apply a mask to the image so you can see where the Mask R-CNN thinks an object is.

[//]: # (Image References)

[image1]: ./output/Figure_1.png "Mask-RCNN output"

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set.

The CelebA dataset consist of:

* 202,599 number of face images of various celebrities
* 10,177 unique identities, but names of identities are not given
* 40 binary attribute annotations per image
* 5 landmark locations

### Model Choice

The Mask R-CNN was trained on the COCO dataset, which has L=90 classes, thus the resulting volume size from the mask module of the Mask R CNN is 100 x 90 x 15 x 15.

Mask R-CNN combined with OpenCV made it very efficient and fast.

### Output of the Model

[image1]

