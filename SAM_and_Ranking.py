import json
import glob
import base64
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pycocotools.mask as maskUtils
from segment_anything import sam_model_registry, SamPredictor


# Variables to be set by user 
imagePaths = glob.glob("example/path/to/Dataset/*.jpg")    # list of strings to each image in the dataset (use glob), all images need to be the same size
amountOfPaths = 1                                           # Amount of fixation paths per image previously computed by DeepGaze
amountOfFixations = 6                                       # Amount of fixations per fixationpath previously computed by DeepGaze
max_amount_of_masks = 0                                     # Maximum amount of returned ranked masks, 0 means no limit

deepGaze_results = json.load(open("DeepGaze_results_" + str(amountOfPaths) + "_paths_" + str(amountOfFixations) + "_fixations.json"))   # Load DeepGaze fixations


def IoU(mask1, mask2):
    i = np.logical_and(mask1, mask2)
    u = np.logical_or(mask1, mask2)
    return np.sum(i) / np.sum(u)

def submask(mask1, mask2):
    return (np.logical_or(mask1, mask2).astype(int) == mask2).all() or (np.logical_or(mask1, mask2).astype(int) == mask1).all()

def combinedLength(list):
    length = 0
    for l in list:
        length += len(l)
    return length


device = 'cuda' # use GPU
sam_checkpoint = "sam_vit_h_4b8939.pth" # TODO: !!!file is not in repo!!! put in readme as requirement
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

results = []

for imagePath in tqdm(imagePaths):
    image = plt.imread(imagePath)

    predictor.set_image(image)

    for result in deepGaze_results:
        if result['file_name'] == imagePath.split("/")[-1]:
            fixationPaths = result['fixationPaths']

    fixationPaths = np.array(fixationPaths)

    masks_list = []
    scores_list = []
    associated_fixations_list = []
    for fixationPath in fixationPaths:
        masks = []
        scores = []
        associated_fixations = []
        for fixation in fixationPath:
            mask, score, _ = predictor.predict(point_coords=np.array([fixation]), point_labels=[1], multimask_output=True)   # Use Multimask

            score1 = []
            mask1 = []

            for i in range(len(mask)):
                if score[i] >= 0.75:
                    score1.append(score[i])
                    mask1.append(mask[i])

            mask_size = []
            if len(mask1) != 0:
                for i in range(len(mask1)):
                    mask_size.append(np.sum(mask1[i]))
            else:
                continue

            
            index = np.argmax(mask_size)
            mask = mask1[index]
            score = score1[index] 

            masks.append(mask)
            scores.append(score)
            associated_fixations.append([fixation])
        
        masks_list.append(masks)
        scores_list.append(scores)
        associated_fixations_list.append(associated_fixations)
                
    
    while True:
        combined_length = combinedLength(masks_list)
        for x, masks in enumerate(masks_list):
            if combined_length != combinedLength(masks_list): break
            for i, mask1 in enumerate(masks):
                if combined_length != combinedLength(masks_list): break
                for j, mask2 in enumerate(masks):
                    if i == j: continue
                    if np.sum(mask1) < 500:                 # filter out to small masks
                        masks_list[x].pop(i)
                        scores_list[x].pop(i)
                        break
                    if np.sum(mask2) < 500:
                        masks_list[x].pop(j)
                        scores_list[x].pop(j)
                        break

                    if (submask(mask1, mask2)) or (IoU(mask1, mask2) >= 0.25):  # if one mask is submask of the other, prompt sam again with the according viewpoints for both masks
                        point_coords = associated_fixations_list[x][i] + associated_fixations_list[x][j]
                        point_labels = np.array([1] * len(point_coords))
                        mask, score, _ = predictor.predict(point_coords= np.array(point_coords), point_labels= point_labels, multimask_output=True)
                        index = np.argmax(np.sum(mask, axis=(1, 2)))            # chose biggest mask again
                        mask = mask[index]
                        score = score[index] 
                        masks_list[x].append(mask)
                        scores_list[x].append(score)
                        associated_fixations_list[x].append(point_coords)
                        masks_list[x].pop(i)
                        masks_list[x].pop(j)
                        scores_list[x].pop(i)
                        scores_list[x].pop(j)
                        associated_fixations_list[x].pop(i)
                        associated_fixations_list[x].pop(j)
                        break
                    
        if combined_length == combinedLength(masks_list): break

    masks_list_length = len(masks_list) 
    masks_list.append([])
    scores_list.append([])
    associated_fixations_list.append([])
    while True: 
        combined_length = combinedLength(masks_list)
        for x, masks1 in enumerate(masks_list):
            if combined_length != combinedLength(masks_list): break
            for y, masks2 in enumerate(masks_list):
                if combined_length != combinedLength(masks_list): break
                if x == y: continue
                masks1_length = len(masks1)
                masks2_length = len(masks2)
                for i, mask1 in enumerate(masks1):
                    if combined_length != combinedLength(masks_list): break
                    for j, mask2 in enumerate(masks2):
                        if (submask(mask1, mask2)) or (IoU(mask1, mask2) >= 0.25):
                            point_coords = associated_fixations_list[x][i] + associated_fixations_list[y][j]
                            point_labels = np.array([1] * len(point_coords))
                            mask, score, _ = predictor.predict(point_coords= np.array(point_coords), point_labels= point_labels, multimask_output=True)
                            index = np.argmax(np.sum(mask, axis=(1, 2)))
                            mask = mask[index]
                            score = score[index] 
                            masks_list[masks_list_length].append(mask)                          # TODO: this has to be fixed
                            scores_list[masks_list_length].append(score)
                            associated_fixations_list[masks_list_length].append(point_coords)
                            masks_list[x].pop(i)
                            masks_list[y].pop(j)
                            scores_list[x].pop(i)
                            scores_list[y].pop(j)
                            associated_fixations_list[x].pop(i)
                            associated_fixations_list[y].pop(j)
                            break

        if combined_length == combinedLength(masks_list): break

    masks = []
    for mask_list in masks_list:
        masks += mask_list

    rankings = []
    for x, fixationPath in enumerate(fixationPaths):
        ranking = [0] * len(masks)
        for i, mask in enumerate(masks):
            for j, fixation in enumerate(fixationPath):
                if mask[int(fixation[1]), int(fixation[0])] == 1:
                    if (ranking[i] == 0) or (ranking[i] > j):
                        if j != len(fixationPath):
                            ranking[i] = 1 - (j / (len(fixationPath)))
                        else:
                            ranking[i] = 0.01        # if fixation is last in path, set ranking to 0.01 (to avoid 0) this way there should be no mask that is not ranked
        rankings.append(ranking)

    ranking = np.mean(rankings, axis=0).tolist()

    if max_amount_of_masks != 0:
        while True:
            if len(masks) > max_amount_of_masks:
                index = np.argmax(ranking)
                masks.pop(index)
                ranking.pop(index)
            else:
                break

    encoded_masks = []
    for mask in masks:
        encoded_masks.append(maskUtils.encode(np.asfortranarray(mask)))
    
    fully_encoded_masks = []
    for mask in encoded_masks:
        mask['counts'] = base64.b64encode(mask['counts']).decode("utf-8")
        fully_encoded_masks.append(mask)

    # Save masks and Rankings to dict
    results.append({"file_name": imagePath.split("/")[-1], "ranking": ranking, "masks": fully_encoded_masks})
    
     
# Save dict to file
with open("SAM_results_" + str(amountOfPaths) + "_paths_" + str(amountOfFixations) + "_fixations.json", "w") as file:
    json.dump(results, file)

print("Success, results saved as SAM_results_" + str(amountOfPaths) + "_paths_" + str(amountOfFixations) + "_fixations.json")
