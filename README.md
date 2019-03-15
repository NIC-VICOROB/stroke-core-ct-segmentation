# stroke-core-ct-segmentation
Docker for training and running the tool for stroke lesion core segmentation presented in our publication.

## Pre-requisites
- Python and Tkinter
- nvidia-docker-2 (https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0))

## Dataset specification file
For accessing the images, a .txt specification file must be placed in the _root directory_ containing the images. The images for a patient are specified each in one line, using _relative_ paths to the _root directory_ of the dataset, with a blank line between the images of different patients.  
* For a correct functioning of the preprocessing the Non Contrast CT image must be placed first.
* For training, the ground truth image (an image with binary intensities) must be placed last. 
* For inference, no ground truth image must be present in the specification file.

For using the ISLES 2018 dataset, two .txt specification files are provided. This are to be placed on the root of the decompressed dataset folder available at https://www.smir.ch/ISLES/Start2018 after registration.

## Running the program
1. Download the repository in .zip or clone using the terminal:
`git clone https://github.com/NIC-VICOROB/stroke-core-ct-segmentation.git`

2. From the root directory of the repository run:
`python gui.py`
