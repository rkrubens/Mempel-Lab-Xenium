#!/bin/bash -l
#SBATCH -o outs/bm_mca_%j.out
#SBATCH -e outs/bm_mca_%j.err
#SBATCH -J bm_mca
#SBATCH -q gpu_normal
#SBATCH -p gpu_p
#SBATCH -c 20
#SBATCH --mem=500G
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00

source /ictstr01/home/icb/raphael.kfuri-rubens/miniforge-pypy3/etc/profile.d/conda.sh
conda activate int-bm-env

python bm_mca.py