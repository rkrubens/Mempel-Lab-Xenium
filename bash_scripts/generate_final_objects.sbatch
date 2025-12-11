#!/bin/bash -l
#SBATCH -J gen_xen_adata
#SBATCH -o outs/gen_xen_adata_%A_%a.out
#SBATCH -e outs/gen_xen_adata_%A_%a.err
#SBATCH -q gpu_normal
#SBATCH -p gpu_p
#SBATCH -c 8
#SBATCH --mem=500G
#SBATCH -t 24:00:00
#SBATCH --array=0-1

source /ictstr01/home/icb/raphael.kfuri-rubens/miniforge-pypy3/etc/profile.d/conda.sh
conda activate int-bm-env

echo "[INFO] Starting job on host: $(hostname)"
echo "[INFO] SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"

srun python /home/icb/raphael.kfuri-rubens/git/mempel_xenium/generate_final_objects.py
