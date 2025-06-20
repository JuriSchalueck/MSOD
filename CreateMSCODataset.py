import json
import glob
import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

cocoSearch18ImagePaths = glob.glob("Resources/images/*/*.jpg")
cocoAnnTrain  = COCO("Resources/annotations_trainval2017/annotations/instances_train2017.json")
cocoAnnVal = COCO("Resources/annotations_trainval2017/annotations/instances_val2017.json")
cocoFreeView = json.load(open("Resources/COCOFreeView_fixations_trainval.json"))
dataset = []


def resize_and_pad_image(img, target_width=1680, target_height=1050):
    img = Image.fromarray(img)
    original_width, original_height = img.size
    target_aspect_ratio = target_width / target_height
    original_aspect_ratio = original_width / original_height

    if original_aspect_ratio > target_aspect_ratio:
        new_width = target_width
        new_height = int(new_width / original_aspect_ratio)
    else:
        new_height = target_height
        new_width = int(new_height * original_aspect_ratio)

    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    new_image.paste(resized_img, (x_offset, y_offset))

    return new_image


# remove image duplicates
reducedImagePaths = []
helper = []
for pth in cocoSearch18ImagePaths:
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

# iterate over remaining (4317) images
reducedImagePaths3 = []
for pth in tqdm.tqdm(reducedImagePaths2):
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

    # TODO: Fix worng resizing here remember there are borders in COCOSearch 18 images so we have to recreate that for the masks
    binaryMasks = []
    for ann in anns: 
        try:
            binaryMasks.append(np.array(resize_and_pad_image(cocoAnnTrain.annToMask(ann)))[:,:,2]) #TODO: find more elegant solution
        except KeyError:
            binaryMasks.append(np.array(resize_and_pad_image(cocoAnnVal.annToMask(ann)))[:,:,2])

    rankings = []
    for viewPth in viewPths:
        voting = [-1] * len(binaryMasks)
        fixations_x = viewPth['X']
        fixations_y = viewPth['Y']
        fixations = np.array([fixations_x, fixations_y]).reshape((len(fixations_x), 2))

        for i, mask in enumerate(binaryMasks):
            for j, fixation in enumerate(fixations): 
                if(fixation[0] < 1050 and fixation[1] < 1680):                  #TODO dosent matter because there are no object masks on that border anymore
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

with open("Resources/MSCO.json", "w") as file:
    json.dump(dataset, file)

for pth in tqdm.tqdm(reducedImagePaths3):
    img = plt.imread(glob.glob("Resources/MSCOCOOriginalImages/*/" + pth.split("/")[-1])[0])

    # if image is single channel (grayscale image) change it to rbg. if not roblem with creating tensors later with DeepGaze and SAM
    if len(img.shape) == 2:
        img = np.stack(([img] * 3), 2)

    plt.imsave("Resources/MSCOImages/" + (pth.split("/")[-1]), img)

print("FINISHED! Images saved at Resources/MSCOImages/ and mask/rankings saved as MSCO.json in Resources/")
