# Multiple Salient Object Detection with DeepGaze III and SAM

This repository contains the code to my Bacheors Thesis in which I presentet MSOD which combines the DeepGaze III model with the foundational SAM model to predict objects and their relative salience in a image.

## Contents

The repository contains multiple files:
- This README.md file
- A requirements.txt file with the necascerry python dependencies
- A python script that combines the COCO dataset with the COCO-FreeView dataset to create our MSCO (Multiple Salient COCO Objects) dataset named CreateMSCODataset.py
- A python script for the Deepgaze part of our proposed model DeepGaze.py
- A python script for the SAM and Ranking part of our model SAM_and_Ranking.py
- Two evaluation scripts, one for our MSCO Dataset and one for the ASSR Dataset
- The SOR.py file for evaluation
- The SASOR.py file for evaluation

## Additional requirements

- The ViT-H SAM model which you can download from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
