#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --time=10:00:00
#SBATCH -o train.log

date

hostname
nvidia-smi

. ~/.bashrc
conda activate blaze
python -u tdnn_lstm_mmi/train.py

date
