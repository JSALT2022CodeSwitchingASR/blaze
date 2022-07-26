#!/bin/bash
env_name=blaze

# If you are behind a proxy write in `~/.pip/pip.config`:
# [global]
# proxy = <your-proxy>
export PIP_CONFIG_FILE=~/.pip/pip.confg

# Create the environment
conda create -n $env_name python=3.8
conda activate $env_name
conda install -c anaconda cmake

# Install pytorch LTS
pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111

# Install lhotse
pip --proxy $http_proxy install git+https://github.com/JSALT2022CodeSwitchingASR/lhotse.git

# Install icefall
pip install git+https://github.com/k2-fsa/icefall.git

# Some more dependencies
pip install \
    more-itertools \
    pyarabic \
    lxml

conda install -c conda-forge libsndfile

# MatGraph
git clone https://github.com/FAST-ASR/matgraph.git
cd matgraph
pip install -r requirements.txt
python setup.py install
cd -
