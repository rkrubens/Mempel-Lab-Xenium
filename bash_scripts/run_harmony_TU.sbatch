#!/bin/bash -l
#SBATCH -o outs/harmony_tu%j.out
#SBATCH -e outs/harmony_tu%j.err
#SBATCH -J harmony_tu
#SBATCH -q gpu_normal
#SBATCH -p gpu_p
#SBATCH -c 20
#SBATCH --mem=500G
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00

source /ictstr01/home/icb/raphael.kfuri-rubens/miniforge-pypy3/etc/profile.d/conda.sh
conda activate harmony-env 

python harmony_integration.py \
  --input /home/icb/raphael.kfuri-rubens/git/mempel_xenium/resegmentation_adatas_merged/merged_TU.h5ad \
  --batch-key slide_id \
  --output /home/icb/raphael.kfuri-rubens/git/mempel_xenium/resegmentation_integration_embs/harmony_TU_emb.pkl
