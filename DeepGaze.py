import json
import torch
import glob
import numpy as np
from tqdm import tqdm
import deepgaze_pytorch
import matplotlib.pyplot as plt
from pysaliency.models import sample_from_logdensity


# Variables to be set by user 
imagePaths = glob.glob("example/path/to/Dataset/*.jpg")    # list of strings to each image in the dataset (use glob), all images need to be the same size
amountOfPaths = 1                                           # Amount of fixation paths generated per image
amountOfFixations = 6                                       # Amount of fixations generated per fixationpath


def get_fixation_history(fixation_coordinates, model):
    history = []
    for index in model.included_fixations:
        try:
            history.append(fixation_coordinates[index])
        except IndexError:
            history.append(np.nan)
    return history


device = 'cuda' # use GPU

## DeepGaze III initialization
deepGaze_model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(device)


# Generate Centerbias
img_shape = plt.imread(imagePaths[0]).shape
x, y = np.meshgrid(np.linspace(-1, 1, img_shape[1]), np.linspace(-1, 1, img_shape[0]))
centerbias = np.exp(-((np.sqrt(x*x + y*y) - 0.0)**2 / (1.0 * 1.0**2)))
centerbias_tensor = torch.tensor(np.array([centerbias])).to(device)


deepGaze_results = []

for imagePath in tqdm(imagePaths):

    image = plt.imread(imagePath)
    image_tensor = torch.tensor([image.transpose(2, 0, 1)]).to(device)
    rst = np.random.RandomState(seed=23)
    fixationPaths = []

    for i in range(amountOfPaths): 
        fixations_x, fixations_y = [img_shape[1]/2], [img_shape[0]/2]

        for j in range(amountOfFixations):
            x_hist = get_fixation_history(fixations_x, deepGaze_model)
            y_hist = get_fixation_history(fixations_y, deepGaze_model)
            
            x_hist_tensor = torch.tensor([x_hist]).to(device)
            y_hist_tensor = torch.tensor([y_hist]).to(device)
            log_density_prediction = deepGaze_model(image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)
            logD = log_density_prediction.detach().cpu().numpy()[0, 0]
            next_x, next_y = sample_from_logdensity(logD, rst=rst)

            fixations_x.append(next_x)
            fixations_y.append(next_y)

        fixationPaths.append(np.array([fixations_x, fixations_y]).T[1:])

    fixationPaths = np.array(fixationPaths).tolist()

    deepGaze_results.append({"file_name": imagePath.split("/")[-1], "fixationPaths": fixationPaths})

# Save to file
with open("DeepGaze_results_" + str(amountOfPaths) +"_paths_" + str(amountOfFixations) +"_fixations.json", "w") as file: #TODO number of fixations into config
    json.dump(deepGaze_results, file)

print("Success, DeepGaze results saved as DeepGaze_results_" + str(amountOfPaths) +"_paths_" + str(amountOfFixations) +"_fixations.json")
