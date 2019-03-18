# Acute ischemic stroke lesion core segmentation in CT perfusion images using fully convolutional neural networks. 
Tool for training and inference for stroke lesion core segmentation as presented in:

Acute ischemic stroke lesion core segmentation in CT perfusion images using fully convolutional neural networks. 
Albert Clèrigues*, Sergi Valverde, Jose Bernal, Jordi Freixenet, Arnau Oliver, Xavier Lladó. Unpublished.

## Pre-requisites
- Python and Tkinter
- nvidia-docker2 (https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0))

## Dataset specification file
For accessing the images, a .txt specification file must be **placed on the root directory of the dataset folder** containing the nifti images (.nii, .nii.gz). The images for a patient are specified each in one line, using **relative_ paths to the _root directory_ of the dataset**, with a blank line between the images of different patients. Then, using the GUI use the 'Browse...' button on the 'dataset file' field to point to the desired dataset file.
* For preprocessing, the Non Contrast _CT_ image must be placed _first_. 
* For skull stripping, the background of all provided modalities must be exactly 0.
* For training, the ground truth image (an image with binary intensities) must be placed last. 
* For inference, no ground truth image must be present in the specification file.

Two example dataset specification files (.txt) are provided for using the ISLES 2018 dataset. This dataset is available at https://www.smir.ch/ISLES/Start2018 after registration. After download, copy the training and testing specification files to the uncompressed folder and then load them using the GUI.

## Running the program
1. Download the repository in .zip or clone using the terminal:
`git clone https://github.com/NIC-VICOROB/stroke-core-ct-segmentation.git`

2. From the root directory of the repository run:
`python gui.py`

## Using the GUI

#### Training

![Training GUI](storage/readme_images/training.png?raw=true "Training GUI")

1. Data
    - Using the 'Browse...' button, navigate to the datset folder and select the dataset specification file.
2. Preprocessing
    - Enable the desired options (check the dataset specification file format is well built before!!!)
3. CNN Model
    - Write a name for referring to the trained network, which then can be further trained or used for inference.

#### Inference

![Inference GUI](storage/readme_images/inference.png?raw=true "Inference GUI")

1. Data
    - Using the 'Browse...' button, navigate to the datset folder and select the dataset specification file.
2. Preprocessing
    - Enable the desired options (check the dataset specification file format is well built before!!!)
3. CNN Model
    - Select a pretrained model to use and **check the dataset specification files for training and inference have the modality images in the same order**.

 ## Results
 The resulting probability maps are stored in a timestamped folder inside the `results` folder.
