import json
import math
import glob
import tqdm
import base64
import tomllib
import numpy as np
from SASOR import evalu
from pathlib import Path
import skimage.color as color
from SOR import sor as sor_eval
import matplotlib.pyplot as plt
from pycocotools import mask as maskUtils

with open("config.toml", "rb") as file:
    toml_data: dict = tomllib.load(file)

Dataset = toml_data['Dataset']
amountOfPaths = toml_data['DeepGaze']['amountOfViewPaths']
amountOfFixations = toml_data['DeepGaze']['amountOfFixations']
pathToASSR = toml_data['Paths']['pathToASSR']
paths = glob.glob(pathToASSR + "rank_order/test/*.json")
imagesPerChunk = toml_data['Evaluation']['imagesPerChunk']

results = json.load(open("Resources/SAM/" + str(Dataset) + "/SAM_results_" + str(amountOfPaths) + "_paths_" + str(amountOfFixations) + "_fixations.json"))


def SASOR(input_data, iou_threshold=.5, name="test"):
    return evalu(input_data, iou_threshold, name)

def SOR(input_data):
    sor_list = []
    for data in input_data:
        #sort masks by ranking
        segmaps = data['segmaps']
        gt_masks = data['gt_masks']
        rank_scores = data['rank_scores']
        gt_ranks = data['gt_ranks']

        sorted_segmaps = []
        sorted_gt_masks = []

        for i in range(len(rank_scores)):
            sorted_segmaps.append(segmaps[np.argmax(rank_scores)])
            rank_scores[np.argmax(rank_scores)] = -1

        for i in range(len(gt_ranks)):
            sorted_gt_masks.append(gt_masks[np.argmax(gt_ranks)])
            gt_ranks[np.argmax(gt_ranks)] = -1

        sor_score = sor_eval(np.array(sorted_segmaps), np.array(sorted_gt_masks))
        sor_list.append(sor_score)

    while True:
        if np.nan in sor_list:
            sor_list.remove(np.nan)
            sor_list.append(0)
        else:
            break

    for i, sor in enumerate(sor_list):
        if math.isnan(sor):
            sor_list[i] = 0

    return np.mean(sor_list)


# this could be calculated using rle encoded masks to save memory
def MAE(input_data):

    def AE(img_data):
        gt_masks = img_data['gt_masks']
        segmaps = img_data['segmaps']
        hight, width = gt_masks[0].shape
        combined_gt_mask = np.zeros((hight, width), dtype=np.bool)
        combined_segmap = np.zeros((hight, width), dtype=np.bool)
        for mask in gt_masks:
            np.logical_or(mask, combined_gt_mask, out=combined_gt_mask)

        for i in range(len(segmaps)):
            np.logical_or(segmaps[i], combined_segmap, out=combined_segmap)

        result = np.sum(np.logical_xor(combined_gt_mask, combined_segmap)) / (hight * width)
        return result
        

    average_errors = []
    for img_data in tqdm.tqdm(input_data):
        average_errors.append(AE(img_data))

    return (np.sum(average_errors)) / len(input_data)


sasor_scores = []
sor_scores = []
mae_scores = []

numberOfChunks = int(len(paths) / imagesPerChunk) + 1
chunks = np.linspace(0, len(paths), numberOfChunks ,dtype=int)
path_chunks = [paths[chunks[i]:chunks[i+1]] for i in range(len(chunks)-1)]

for z, path_chunk in tqdm.tqdm(enumerate(path_chunks)):
    print("Chunk: ", z)
    input_data = []
    for x, pth in tqdm.tqdm(enumerate(path_chunk)):
        img_data = {}
        gt_masks = []
        segmaps = np.array([])
        gt_ranks = []

        imgName = pth.split("/")[-1].split(".")[0] + ".jpg" # somehow double .jpg ending but dosent work otherwise so fix later

        gt = json.load(open(pth))
        # gt_masks
        imgID = gt['image_id']
        original_masks = color.rgb2gray(plt.imread(pathToASSR + 'gt/test/' + str(imgID) +'.png'))
        unique_values = np.unique(original_masks)

        for i in range(1, len(unique_values)):
            mask = np.where(original_masks == unique_values[i], 1, 0)
            gt_masks.append(mask)

        # gt_ranks
        gt_ranks = np.linspace(0, 1,len(unique_values))
        gt_ranks = gt_ranks[1:]

        # segmaps / rank_scores
        for result in results:
            if result['file_name'] == imgName:
                rank_scores = result['ranking']
                binMasks = []
                for mask in result['masks']:
                    mask["counts"] = base64.b64decode(mask["counts"])
                    binMasks.append(maskUtils.decode(mask))
                segmaps = np.array(binMasks)

        img_data['gt_masks'] = gt_masks
        img_data['gt_ranks'] = gt_ranks
        img_data['segmaps'] = segmaps
        img_data['rank_scores'] = rank_scores

        input_data.append(img_data)

    print("Calculating SA_SOR Score:")
    sasor_result = SASOR(input_data)
    sasor_scores.append(sasor_result)
    print("SA-SOR Score: ", sasor_result)
    print("Calculating SOR Score:")
    sor_result = SOR(input_data)
    sor_scores.append(sor_result)
    print("SOR Score: ", sor_result)
    print("Calculating MAE Score:")
    mae_result = MAE(input_data)
    mae_scores.append(mae_result)
    print("MAE Score: ", mae_result)


mean_sasor = np.mean(sasor_scores)
mean_sor = np.mean(sor_scores)
mean_mae = np.mean(mae_scores)

print("final SASOR Score: ", np.mean(sasor_scores))
print("final SOR Score: ", np.mean(sor_scores))
print("final MAE Score: ", np.mean(mae_scores))

#save Evaluation results to file
Path("Resources/Results/" + str(Dataset)).mkdir(parents=True, exist_ok=True)
with open("Resources/Results/" + str(Dataset) + "/Eval_results_" + str(Dataset) + "_" + str(amountOfPaths) +"_paths_" + str(amountOfFixations) + "_fixations.json", "w") as file:
    json.dump({"SASOR": sasor_scores, "MAE": mae_scores, "SOR": sor_scores, "MAE": mae_scores, "combined_SASOR": mean_sasor,"combined_SOR": mean_sor, "combined_MAE": mean_mae}, file)

print("Results succesfully saved in Resources/Results/" + str(Dataset) + "/ as Eval_results_" + str(Dataset) + "_" + str(amountOfPaths) +"_paths_" + str(amountOfFixations) + "_fixations.json")
