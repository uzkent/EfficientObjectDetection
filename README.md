# Efficient Object Detection in Large Images with Deep Reinforcement Learning

This repository contains PyTorch implementation of our IEEE WACV20 paper on Efficient Object Detection in Large
Images with Deep Reinforcement Learning. The arxiv version of the paper can be found [here](https://arxiv.org/abs/1912.03966).

## Object Detectors
### Training Object Detectors
We train two different detectors: (1) **Fine level** object detector and (2) **Coarse level** object detector. To parameterize
the detectors, we use the YOLO-v3 network, however, you can use a detector of your choice. We use [this repository](https://github.com/eriklindernoren/PyTorch-YOLOv3) to train the
fine and coarse level detectors. In the paper, we use 320x320px and 64x64px images to train the fine and coarse level detectors. The coarse images represented with 64x64 px is the downsampled version of the high resolution images represented with 320x320 px. Each image ideally should represent a window of the large images that we process for object detection.


### Testing Object Detectors
Once we train the object detectors, we run them on the training and testing images for the Policy Network. **The large training images for the Policy Network should be different than the training images for the object detectors**.
