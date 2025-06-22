# Multiple Salient Object Detection with DeepGaze III and SAM

This repository contains the UPDATED code to my Bachelor's Thesis project where I presented MSOD. MSOD combines the DeepGaze III model with the foundational SAM model to predict objects and their relative salience in an image.

## Contents

The repository contains multiple files:
- README.md
- requirements.txt with the python libraries we used to help you set up
- Example_config.toml file which you can copy and rename to create your own configuration file
- MSCOSetup.sh script that downloads and sets up the folders for creating our MSCO Dataset
- MSODSetup.sh script that downloads the ViT-H SAM model and sets up a virtual Python environment
- CreateMSCODataset.py script that generates our proposed MSCO (Multiple Salient COCO Objects) Dataset from COCO images and COCOFreeView Fixation data
- DeepGaze.py script that uses DeepGaze III to predict human-like view paths for the given images
- SAM_and_Ranking.py script that uses SAM and the predicted view paths from the DeepGaze.py script to generate object masks and rank them by their saliency
- Two evaluation scripts Evaluation_MSCO.py and Evaluation_ASSR.py for their respective dataset
- Two helper scripts SOR.py and SASOR.py that help with the evaluation
- .gitignore file

## Getting started

Here we will guide you through setting up a virtual environment to run our scripts and download the SAM model.

### Setting up the virtual environment
You can either execute the MSODSetup.sh script or go through the following steps manually

Navigate to the MSOD folder
```
cd path/to/MSOD
```
then create the environment
```
python3 -m venv .venv
```
activate the environment
```
source .venv/bin/activate
```
and install the requirements
```
pip install -r requirements.txt
```

### Download the ViT-H SAM model

Go to https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints and download the ViT-H SAM model to the MSOD folder

## Choose a Dataset

You can either use one of the two Datasets we have already evaluated on or use your own dataset.

### MSCO

Unfortunately we cannot provide a download to our MSCO Dataset due to complicated licensing of the COCO images. If you want to use our MSCO (Multiple Salient COCO Objects) Dataset you can recreate it yourself. You can either run the MSCOSetup.sh script or do the following steps manually:

- Download the MSCOCO 2017 train and validation images and annotations https://cocodataset.org/#download
- The COCO-FreeView data https://sites.google.com/view/cocosearch/coco-freeview
- And the COCO-Search18 Images both TP and AP https://sites.google.com/view/cocosearch/

Store these in a folder named "Resources" inside of the MSOD folder. Then you unzip all the zipped files within the folder
```
unzip '*.zip'
```
combine the TA and TP images
```
mv COCOSearch18-images-TP/images/ images

cp -R COCOSearch18-images-TA/coco_search18_images_TA/* images
```
and combine the train2017 and val2017 images
```
cp val2017/* train2017
```
If you want to save disk space you can delete the zipfiles and folders that are not used anymore
```
rm -r *.zip COCOSearch18-images-TA COCOSearch18-images-TP val2017
```
**CAUTION!** After running the CreateMSCODataset.py script you can also remove the remaining files
```
rm -r train2017 annotations_trainval2017 images COCOFreeView_fixations_trainval.json
```

The config file is set up to run with the MSCO Dataset by default, it might still be useful to review it and change some values to your liking.

### ASSR

Download the ASSR Dataset from https://github.com/SirisAvishek/Attention_Shift_Ranks/tree/master/Attention_Shift_Saliency_Rank and place it into the Resources folder and then unzip it. In the config file set the pathToImages value to the testing images from ASSR (we do not train our model and to have a fair comparison to others that evaluate using the ASSR dataset we only use the test images), and change the maxAmountOfMasks to 5.

### Your own

You can use your own datasets Simply change the name of the dataset and path to the images in the config file, if you want to set an upper limit for the number of masks you want to generate and you're ready to go.
