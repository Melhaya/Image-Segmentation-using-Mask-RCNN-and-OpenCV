# USAGE
# python mask_rcnn.py --mask-rcnn mask-rcnn-coco --image images/example_01.jpg

# import the necessary packages
import numpy as np
import argparse
import random
import cv2
import os
from matplotlib import pyplot as plt

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-m", "--mask-rcnn", required=True,
	help="base path to mask-rcnn directory")
ap.add_argument("-c", "--confidence", type=float, default=0.8,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="minimum threshold for pixel-wise mask segmentation")
args = vars(ap.parse_args())

# load the COCO class labels the Mask R-CNN was trained on
labelsPath = os.path.sep.join([args["mask_rcnn"],
	"object_detection_classes_coco.txt"])
LABELS = open(labelsPath).read().strip().split("\n")

# Using Red color that will be used when visualizing a given
# instance segmentation
color = np.array([np.array([0,0,255])], dtype="uint8")

# derive the paths to the Mask R-CNN weights and model configuration
weightsPath = os.path.sep.join([args["mask_rcnn"],
	"frozen_inference_graph.pb"])
configPath = os.path.sep.join([args["mask_rcnn"],
	"mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

# load the Mask R-CNN trained on the COCO dataset (90 classes)
# from disk, However it will be used only to show detected persons/faces
print("[INFO] loading Mask R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

# load the input image and grab its spatial dimensions
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

# construct a blob from the input image and then perform a forward
# pass of the Mask R-CNN, giving (1) the bounding box coordinates
# of the objects in the image along with (2) the pixel-wise segmentation
# for each specific object
blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
net.setInput(blob)
(boxes, masks) = net.forward(["detection_out_final", "detection_masks"])

# show volume information on Mask R-CNN
print("[INFO] boxes shape: {}".format(boxes.shape))
print("[INFO] masks shape: {}".format(masks.shape))

# loop over the number of detected objects
for i in range(0, boxes.shape[2]):
	# extract the class ID of the detection along with the confidence
	# associated with the prediction
	classID = int(boxes[0, 0, i, 1])
	confidence = boxes[0, 0, i, 2]

	if LABELS[classID] == "person":

		# filter out weak predictions by ensuring the detected probability
		# is greater than the minimum probability
		if confidence > args["confidence"]:
			# clone the original image so we can draw on it
			clone = image.copy()

			# scale the bounding box coordinates back relative to the
			# size of the image and then compute the width and the height
			# of the bounding box
			box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
			(startX, startY, endX, endY) = box.astype("int")
			boxW = endX - startX
			boxH = endY - startY

			# extract the pixel-wise segmentation for the object, resize
			# the mask such that it's the same dimensions of the bounding
			# box, and then finally threshold to create a *binary* mask
			mask = masks[i, classID]
			mask = cv2.resize(mask, (boxW, boxH),
				interpolation=cv2.INTER_NEAREST)
			mask = (mask > args["threshold"])

			# extract the ROI of the image
			roi = clone[startY:endY, startX:endX]

			# convert the mask from a boolean to an integer mask with
			# to values: 0 or 255, then apply the mask
			visMask = (mask * 255).astype("uint8")
			instance = cv2.bitwise_and(roi, roi, mask=visMask)

			# show the extracted ROI, the mask, along with the
			# segmented instance
			plt.subplot(1, 3, 1).title.set_text("ROI"), plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
			plt.subplot(1, 3, 2).title.set_text("Mask"), plt.imshow(visMask, 'gray')
			plt.subplot(1, 3, 3).title.set_text("Segmented"), plt.imshow(cv2.cvtColor(instance, cv2.COLOR_BGR2RGB))			
			plt.show()