#!/bin/bash -l
#SBATCH -o outs/scvi_tu%j.out
#SBATCH -e outs/scvi_tu%j.err
#SBATCH -J scvi_tu
#SBATCH -q gpu_normal
#SBATCH -p gpu_p
#SBATCH -c 20
#SBATCH --mem=500G
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00

source /ictstr01/home/icb/raphael.kfuri-rubens/miniforge-pypy3/etc/profile.d/conda.sh
conda activate scvi-bm-env

python scvi_integration.py \
  --input /home/icb/raphael.kfuri-rubens/git/mempel_xenium/resegmentation_adatas_merged/merged_TU.h5ad \
  --layer counts \
  --batch-key slide_id \
  --max-epochs 200 \
  --output /home/icb/raphael.kfuri-rubens/git/mempel_xenium/resegmentation_integration_embs/scvi_TU_emb.pkl
