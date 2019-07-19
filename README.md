# Vehicle-Instance-Segmentation-From-Aerial-Image-and-Video
Python implementation of Convolutional Neural Network (CNN) proposed in academia

This repository includes functions to preprocess the input images and their respective polygons so as to create the input image patches, 
mask patches and edge patches to be used for model training. The CNN used here is the Multitask Learning Residual Fully Convolutional 
Network (MLRFCN) model implemented in the paper 
'Vehicle Instance Segmentation From Aerial Image and Video Using a Multitask Learning Residual Fully Convolutional Network' by Mou L., 
Zhu X. X. (2018).


The main differences between the implementations in the paper and the implementation in this repository is as follows:

- Group Normalization is used instead of Batch Normalization, since it is envisaged that very small batch sizes would be used for training 
  this model with consumer - level Graphics Processing Unit (GPU) in view of memory constraints, and it has been shown in academia that 
  Group Normalization outperforms Batch Normalization for very small batch sizes.
  

The group normalization implementation in Keras used in this repository is the exact same class object defined in the group_norm.py file 
located in titu1994's Keras-Group-Normalization github repository at https://github.com/titu1994/Keras-Group-Normalization. 
Please ensure that the group_norm.py file is placed in the correct directory before use.


Requirements:

- cv2
- glob
- json
- numpy
- rasterio
- group_norm (downloaded from https://github.com/titu1994/Keras-Group-Normalization)
- keras (tensorflow backend)
