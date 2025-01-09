import numpy as np
import scipy.stats as stats

def IoU(mask1, mask2):
    i = np.logical_and(mask1, mask2)
    u = np.logical_or(mask1, mask2)
    return np.sum(i) / np.sum(u)

def sor(preds, gts, thres=.5):
    gt_ranks = []
    pred_ranks = []
    n_gt = len(gts)
    n_pred = len(preds)
    for i in range(n_gt):
        for j in range(n_pred):
            iou = IoU(preds[j], gts[i])
            if iou > thres:
                gt_ranks.append(n_gt - i)
                pred_ranks.append(n_pred - j)
                break
    if len(gt_ranks) > 1:
        try:
            spr = stats.spearmanr(pred_ranks, gt_ranks).statistic
        except:
            spr = stats.spearmanr(pred_ranks, gt_ranks).correlation
        return (spr + 1.0)/2.0
    elif len(gt_ranks) == 1:
        return 1.0
    else:
        return np.nan