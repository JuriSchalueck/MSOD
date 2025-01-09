import cv2
import json
import glob
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

# Paths to Datasets
pthToCOCO18Images = "../COCOSearch18"
pthToCOCOAnnTrain = "../Datasets/annotations/instances_train2017.json"
pthToCOCOAnnVal = "../Datasets/annotations/instances_val2017.json"


imagePaths = glob.glob(pthToCOCO18Images + "/*/*/*.jpg")
cocoAnnTrain  = COCO(pthToCOCOAnnTrain)
cocoAnnVal = COCO(pthToCOCOAnnVal)
cocoFreeView = json.load(open("../Datasets/COCOFreeView_fixations_trainval.json"))

dataset = []

# remove image duplicates
reducedImagePaths = []
helper = []

for pth in imagePaths:
    if pth.split("/")[-1] not in helper:
        helper.append(pth.split("/")[-1])
        reducedImagePaths.append(pth)


# remove images with missing masks or fixations
reducedImagePaths2 = []
for pth in reducedImagePaths:
    imgName = pth.split("/")[-1]
    img_id = imgName.split(".")[0].lstrip("0")

    ann_ids = cocoAnnTrain.getAnnIds(imgIds=[int(img_id)], iscrowd=None)
    ann_ids += cocoAnnVal.getAnnIds(imgIds=[int(img_id)], iscrowd=None)

    viewPths = []
    for viewPth in cocoFreeView:
        if viewPth['name'] == imgName:
            viewPths.append(viewPth)

    if ann_ids != [] and viewPths != []:
        reducedImagePaths2.append(pth)

with open("reducedPaths.json", "w") as file:
    json.dump(reducedImagePaths2, file)

lenPaths = len(reducedImagePaths2)

reducedImagePaths3 = []

# iterate over remaining (4317) images
for counter, pth in tqdm(enumerate(reducedImagePaths2)):
    imgName = pth.split("/")[-1]
    img_id = imgName.split(".")[0].lstrip("0")

    ann_ids = cocoAnnTrain.getAnnIds(imgIds=[int(img_id)], iscrowd=None)
    ann_ids += cocoAnnVal.getAnnIds(imgIds=[int(img_id)], iscrowd=None)

    viewPths = []
    for viewPth in cocoFreeView:
        if viewPth['name'] == imgName:
            viewPths.append(viewPth)

    try:
        anns = cocoAnnTrain.loadAnns(ann_ids)
    except KeyError:
        anns = cocoAnnVal.loadAnns(ann_ids)


    binaryMasks = []
    for ann in anns: 
        try:
            binaryMasks.append(cv2.resize(cocoAnnTrain.annToMask(ann), (1680, 1050)))
        except KeyError:
            binaryMasks.append(cv2.resize(cocoAnnVal.annToMask(ann), (1680, 1050)))


    rankings = []
    for viewPth in viewPths:

        voting = [-1] * len(binaryMasks)

        fixations_x = viewPth['X']
        fixations_y = viewPth['Y']

        fixations = np.array([fixations_x, fixations_y]).reshape((len(fixations_x), 2))

        for i, mask in enumerate(binaryMasks):

            for j, fixation in enumerate(fixations):
                    
                if(fixation[0] < 1050 and fixation[1] < 1680):

                    if mask[int(fixation[0]), int(fixation[1])] == 1:
                        
                        if voting[i] == -1:
                            voting[i] = j
        

        norm_voting = np.array(voting) / np.max(voting)

        ranking = [0] * len(binaryMasks)
        for i in range(len(binaryMasks)):
            if norm_voting[i] >= 0:
                ranking[i] = 1 - norm_voting[i]
            else:
                ranking[i] = 0
            
        rankings.append(ranking)
        
    final_ranking = np.mean(np.array(rankings), axis=0)
    final_ranking = final_ranking / np.max(final_ranking) #renormalize ranking
    
    # Remove Masks that did not receive any votes
    final_ranking_copy = final_ranking.copy()
    index_list = []
    for i, ranking in enumerate(final_ranking_copy):
        if ranking == 0.:
            index_list.append(i)
            
    final_ranking = np.delete(final_ranking, index_list)
    binaryMasks = np.delete(binaryMasks, index_list)

    for index in sorted(index_list, reverse=True):
        del anns[index]

    # Add image height and width to the annotation
    img_width = 0
    img_height = 0


    for img in cocoAnnTrain.dataset['images']:
        if img['id'] == int(img_id):
            img_width = img['width']
            img_height = img['height']
            break

    if img_width == 0 or img_height == 0: 
        for img in cocoAnnVal.dataset['images']:
            if img['id'] == int(img_id):
                img_width = img['width']
                img_height = img['height']
                break

    #remove images with Nan values in ranking
    if np.isnan(final_ranking).any():
        continue

    dataset.append({'file_name' : imgName, 'img_id' : img_id, 'img_width' : img_width, 'img_height' : img_height, 'ranking' : final_ranking.tolist(), 'annotations' : anns})        # TODO: remove 'annotations' from dataset they can later be matched with the image id/ image name

    reducedImagePaths3.append(pth)


with open("reducedPaths.json", "w") as file:
    json.dump(reducedImagePaths3, file)

with open("MSCO.json", "w") as file:
    json.dump(dataset, file)

print("FINISHED! Dataset saved as MSCO.json")
