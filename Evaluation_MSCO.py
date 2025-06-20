import json
import math
import glob
import tqdm
import base64
import tomllib
import numpy as np
from SASOR import evalu
from pycocotools import mask as maskUtils
from SOR import sor as sor_eval

with open("config.toml", "rb") as file:
    toml_data: dict = tomllib.load(file)

amountOfPaths = toml_data['DeepGaze']['amountOfViewPaths']
amountOfFixations = toml_data['DeepGaze']['amountOfFixations']
groundTruths = json.load(open(toml_data['Paths']['pathToGroundTruth']))
image_paths = glob.glob(toml_data['Paths']['pathToImages'])
results = json.load(open("Datasets/SAM/SAM_results_" + str(amountOfPaths) + "_paths_" + str(amountOfFixations) + "_fixations.json")) 
imagesPerChunk = toml_data['Evaluation']['imagesPerChunk']

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

numberOfChunks = int(len(image_paths) / imagesPerChunk) + 1
chunks = np.linspace(0, len(image_paths), numberOfChunks ,dtype=int)
path_chunks = [image_paths[chunks[i]:chunks[i+1]] for i in range(len(chunks)-1)]

for z, path_chunk in tqdm.tqdm(enumerate(path_chunks)):
    print("Chunk: ", z)
    input_data = []
    for x, pth in tqdm.tqdm(enumerate(path_chunk))
        img_data = {}
        gt_masks = []
        segmaps = np.array([])
        gt_ranks = []
        rank_scores = []

        imgName = pth.split("/")[-1]
        img_id = imgName.split(".")[0].lstrip("0")

        # gt_masks this causes "long" compute time
        for gt in groundTruths:
            if gt['img_id'] == img_id:
                for ann in gt['annotations']:
                    if type(ann['segmentation']) == list:
                        # polygon
                        binMask = np.zeros((gt['img_height'], gt['img_width']))
                        for seg in ann['segmentation']:
                            rle = maskUtils.decode(maskUtils.frPyObjects([seg], gt['img_height'], gt['img_width']))
                            binMask = binMask + rle[:,:,0]
                        gt_masks.append(binMask.squeeze())
                    else:
                        if type(ann['segmentation']['counts']) == list:
                            rle = maskUtils.frPyObjects([ann['segmentation']], gt['img_height'], gt['img_width'])
                        else:
                            rle = [ann['segmentation']]
                        gt_masks.append(maskUtils.decode(rle).squeeze())

        # gt_ranks
        for gt in groundTruths:
            if gt['img_id'] == img_id:
                for rank in gt['ranking']:
                    if math.isnan(rank):
                        gt_ranks.append(0)
                        continue
                    gt_ranks.append(int(rank * 100))                                           


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
with open("Datasets/Results/MSCOResults/Eval_results_MSCO_" + str(amountOfPaths) +"_paths_" + str(amountOfFixations) + "_fixations.json", "w") as file:
    json.dump({"SASOR": sasor_scores, "SOR": sor_scores, "MAE": mae_scores, "combined_SASOR": mean_sasor,"combined_SOR": mean_sor, "combined_MAE": mean_mae}, file)

print("Results succesfully saved in Datasets/Results/MSCOResults/ as Eval_results_MSCO_" + str(amountOfPaths) +"_paths_" + str(amountOfFixations) + "_fixations.json")
