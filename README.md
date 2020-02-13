# Efficient Object Detection in Large Images with Deep Reinforcement Learning

This repository contains PyTorch implementation of our IEEE WACV20 paper on Efficient Object Detection in Large
Images with Deep Reinforcement Learning. The arxiv version of the paper can be found [here](https://arxiv.org/abs/1912.03966).

## Object Detectors
### Training Object Detectors
We train two different detectors: (1) **Fine level** object detector and (2) **Coarse level** object detector. To parameterize
the detectors, we use the YOLO-v3 network, however, you can use a detector of your choice. We use [this repository](https://github.com/eriklindernoren/PyTorch-YOLOv3) to train the
fine and coarse level detectors. In the paper, we use 320x320px and 64x64px images to train the fine and coarse level detectors. The coarse images represented with 64x64 px is the downsampled version of the high resolution images represented with 320x320 px. Each image ideally should represent a window of the large images that we process for object detection.


### Testing Object Detectors
Once we train the object detectors, we run them on the training and testing images for the Policy Network. **The large training images for the Policy Network should be different than the training images for the object detectors since the object detectors perform much better on the seen images**. Next, we run the coarse and fine level detectors on the windows from the large images and save the normalized bounding boxes ([0,1]) into a numpy array as:
  ``
  x_center, y_center, width, height, objectness score, class
  ...
  ``
Each numpy array needs to be saved in the following format:
  ``
  '{}_{}_{}.npy'.format(image_name, x index of the window, y index of the window)
  ``
The numpy arrays for the fine and coarse level detectors should be saved into two different folders.

Additionally, we save the mAR values for each window of the large image as follows:
  ``
  mAR for window 1,1 ... ... ... mAR for window 1,N
  ...
  ...
  mAR for Window N,1 ... ... ... mAR for window N,N
  ``
The matrix is then saved into a numpy file in the following format: ``'{}{}.format(image_name, detector_type)'``. If your application prioritizes precision over recall, you can use mAP values to train the Policy Network.

## Training the Policy Network
