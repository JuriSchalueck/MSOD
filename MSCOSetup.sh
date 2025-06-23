#! /bin/bash
# This script downloads all the required Data to create our MSCO Dataset and 
# sets them up in the correct foldet structure
# and deletes the files that are not used anymore after

pthToResources="Resources"
pthToTrain2017="train2017.zip"
pthToVal2017="val2017.zip"
pthToAnnotations="annotations_trainval2017.zip"
pthToTPImages="COCOSearch18-images-TP.zip"
pthToTAImages="COCOSearch18-images-TA.zip"
pthToFixations="COCOFreeView_fixations_trainval.json"

#Check if Resources folder exists
if ! [ -d "$pthToResources" ]; then
    mkdir Resources
fi
cd Resources

if ! [ -f "$pthToTrain2017" ]; then
    wget http://images.cocodataset.org/zips/train2017.zip
fi
unzip train2017.zip

if ! [ -f "$pthToVal2017" ]; then
    wget http://images.cocodataset.org/zips/val2017.zip
fi
unzip val2017.zip
cp val2017/* train2017

if ! [ -f "$pthToAnnotations" ]; then
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
fi
unzip annotations_trainval2017.zip 

if ! [ -f "$pthToTPImages" ]; then
    wget http://vision.cs.stonybrook.edu/~cvlab_download/COCOSearch18-images-TP.zip
fi
unzip COCOSearch18-images-TP.zip

if ! [ -f "$pthToTAImages" ]; then
    wget http://vision.cs.stonybrook.edu/~cvlab_download/COCOSearch18-images-TA.zip
fi
unzip COCOSearch18-images-TA.zip
mv COCOSearch18-images-TP/images/ images
cp -R COCOSearch18-images-TA/coco_search18_images_TA/* images

if ! [ -f "$pthToFixations" ]; then
    wget http://vision.cs.stonybrook.edu/~cvlab_download/COCOFreeView_fixations_trainval.json
fi

rm -r *.zip coco_search18_images_TA val2017 __MACOSX
