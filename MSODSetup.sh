#! /bin/bash
# This script does ...

pthToSAMModel="sam2.1_hiera_large.pt"
pthToSAMModelConfig="sam2.1_hiera_l.yaml"
pthToVenv=".venv"

if ! [ -d "$pthToVenv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install -r requirements.txt

if ! [ -f "$pthToSAMModel" ]; then
    wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
fi

if ! [ -f "$pthToSAMModelConfig" ]; then
    wget https://github.com/facebookresearch/sam2/blob/main/sam2/configs/sam2.1/sam2.1_hiera_l.yaml
fi