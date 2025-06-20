import json
import glob
import torch
import tomllib
import numpy as np
from tqdm import tqdm
import deepgaze_pytorch
from pathlib import Path
import matplotlib.pyplot as plt
from pysaliency.models import sample_from_logdensity

# Load config
with open('config.toml', 'rb') as file:
    toml_data: dict = tomllib.load(file)

Dataset = toml_data['Dataset']
imagePaths = glob.glob(toml_data['Paths']['pathToImages'])
amountOfPaths = toml_data['DeepGaze']['amountOfViewPaths']
amountOfFixations = toml_data['DeepGaze']['amountOfFixations']

# DeepGaze Model initialization
device = 'cuda' # use GPU
deepGaze_model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(device)

def get_fixation_history(fixation_coordinates, model):
    history = []
    for index in model.included_fixations:
        try:
            history.append(fixation_coordinates[index])
        except IndexError:
            history.append(np.nan)
    return history

deepGaze_results = []
for imagePath in tqdm(imagePaths):
    image = plt.imread(imagePath) # Load image

    # Generate Centerbias
    x, y = np.meshgrid(np.linspace(-1, 1, image.shape[1]), np.linspace(-1, 1, image.shape[0]))
    centerbias = np.exp(-((np.sqrt(x*x + y*y) - 0.0)**2 / (1.0 * 1.0**2)))
    centerbias_tensor = torch.tensor(np.array([centerbias])).to(device)
    
    image_tensor = torch.tensor([image.transpose(2, 0, 1)]).to(device)
    rst = np.random.RandomState(seed=23)

    fixationPaths = []
    for i in range(amountOfPaths): 
        fixations_x, fixations_y = [image.shape[1]/2], [image.shape[0]/2]

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
Path("Resources/DeepGaze/" + str(Dataset)).mkdir(parents=True, exist_ok=True)
with open("Resources/DeepGaze/" + str(Dataset) + "/DeepGaze_results_" + str(amountOfPaths) +"_paths_" + str(amountOfFixations) +"_fixations.json", "w") as file:
    json.dump(deepGaze_results, file)

print("Success, DeepGaze results saved as DeepGaze_results_" + str(amountOfPaths) +"_paths_" + str(amountOfFixations) +"_fixations.json in Resources/DeepGaze/" + str(Dataset) + "/")
