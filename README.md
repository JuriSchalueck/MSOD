# Multiple Salient Object Detection with DeepGaze III and SAM

This repository contains the UPDATED code to my Bacheor's Thesis in which I presented MSOD. MSOD combines the DeepGaze III model with the foundational SAM model to predict objects and their relative salience in an image.

## Contents

The repository contains multiple files:
- This README.md file
- A requirements.txt file with the necessary python dependencies
- A python script that combines the COCO Dataset with the COCO-FreeView Dataset to create our MSCO (Multiple Salient COCO Objects) Dataset named CreateMSCODataset.py
- A python script for the Deepgaze part of our proposed model DeepGaze.py
- A python script for the SAM and Ranking part of our model SAM_and_Ranking.py
- Two evaluation scripts, one for our MSCO Dataset and one for the ASSR Dataset
- The SOR.py file which is part of the evaluation
- The SASOR.py file which is also part of the evaluation

## Additional requirements

To get you going, you will need some more things:

- The ViT-H SAM model which you can download from https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints This needs to be in the same directory as the python file.

### MSCO Dataset

Unfortunately we can not provide a download to our MSCO Dataset due to the complicated licensing of COCO images. If you want to use our MSCO (Multiple Salient COCO Objects) Dataset you can recreate it yourself. For that you will need:
- The MSCOCO 2017 Train and val images and annotations https://cocodataset.org/#download
- The COCOFreeView Data https://sites.google.com/view/cocosearch/coco-freeview
- And The COCO-Search18 Images both TP and AP https://sites.google.com/view/cocosearch/

Store these in a directory named "Resources" in the repository.
- Unzip the COCOSearch18-images TP and TA
- Then you have to combine them with 
```
mv COCOSearch18-images-TA/coco_search18_images_TA/ images/

mv COCOSearch18-images-TP/images/ images/
```
then you can delete the remaining files with
```
rm -r COCOSearch18-images-TA COCOSearch18-images-TP COCOSearch18-images-TA.zip COCOSearch18-images-TP.zip
```
and finally unzip the annotations_trainval2017.zip and you are done.

### ASSR Dataset

If you want to do something with the ASSR Dataset, you can download it from https://github.com/SirisAvishek/Attention_Shift_Ranks/tree/master/Attention_Shift_Saliency_Rank

## Recommended Setup


**Most file system actions are based on relative paths!**

Navigate to MSCO folder and setup venv

```
cd path/to/MSCO
python3 -m venv .venv
```
activate the virtual environment and install dependencies
```
source .venv/bin/activate
pip install -r /path/to/requirements.txt
```

## The Example_config file
You can copy the Example_config.toml file and rename it config.toml. Here you can specify where your images are stored, where datasets are stored, and set the parameters you want MSOD to run with. Each option is explained by the comment next to it.