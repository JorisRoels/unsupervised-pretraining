# Unsupervised pretraining for U-Net based EM segmentation

This code provides a PyTorch implementation for unsupervised autoencoder pretraing for U-Net [1] based EM segmentation. 

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. Lecture Notes in Computer Science (Including Subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 9351, 234–241. https://doi.org/10.1007/978-3-319-24574-4_28

## Requirements
- Tested with Python 3.6
- Required Python libraries (these can be installed with `pip install -r requirements.txt`): 
  - torch
  - torchvision
  - numpy
  - tifffile
  - imgaug
  - scipy
  - scikit-image
  - progressbar2 (optional)
  - tensorboardX (optional)
  - jupyter (optional)
- Required data: 
  - [EPFL mitochondria dataset](https://cvlab.epfl.ch/data/data-em/)

## Usage
We provide a [notebook](unet.ipynb) that illustrates data loading, network training and validation. Note that the data path might be different, depending on where you downloaded the EPFL data. 
