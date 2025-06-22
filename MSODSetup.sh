#! /bin/bash
# This script does ...

pthToSAMModel="sam_vit_h_4b8939.pth"
pthToVenv=".venv"

if ! [ -d "$pthToVenv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install -r requirements.txt

if ! [ -f "$pthToSAMModel" ]; then
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
fi
