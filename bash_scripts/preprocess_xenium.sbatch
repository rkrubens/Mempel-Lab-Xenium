#!/bin/bash -l
#SBATCH -o outs/xenium_resegmentation_%j.out
#SBATCH -e outs/xenium_resegmentation_%j.err
#SBATCH -J xenium_resegmentation
#SBATCH -q gpu_normal
#SBATCH -p gpu_p
#SBATCH -c 20
#SBATCH --mem=500G
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00

source /ictstr01/home/icb/raphael.kfuri-rubens/miniforge-pypy3/etc/profile.d/conda.sh
mamba activate spatial-env

python preprocess_xenium.py
