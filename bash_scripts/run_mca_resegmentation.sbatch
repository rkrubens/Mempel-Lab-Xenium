#!/bin/bash -l
#SBATCH -o "outs/xenium_resegment_%A_%a.out"
#SBATCH -e "outs/xenium_resegment_%A_%a.err"
#SBATCH -J xenium_resegment
#SBATCH -q cpu_normal
#SBATCH -p cpu_p
#SBATCH -c 20
#SBATCH --mem=500GB
#SBATCH -t 24:00:00
#SBATCH --array=0-5

BASE_DIR="/ictstr01/groups/ml01/projects/2025_xenium_mempel_lab_raphael.kfuri-rubens/MCA205/resegmentation_data"


mapfile -t slides < <(find "$BASE_DIR" -maxdepth 1 -mindepth 1 -type d -name "output*")

if [ ${#slides[@]} -eq 0 ]; then
    echo "[ERROR] No 'output*' directories found in $BASE_DIR"
    exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#slides[@]}" ]; then
    echo "[INFO] SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID has no corresponding slide (only ${#slides[@]} slides). Exiting."
    exit 0
fi

SLIDE_DIR="${slides[$SLURM_ARRAY_TASK_ID]}"
SLIDE_NAME="$(basename "$SLIDE_DIR")"

echo "[INFO] SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"
echo "[INFO] Processing slide directory: $SLIDE_DIR"
echo "[INFO] Slide name: $SLIDE_NAME"

CORES=${SLURM_CPUS_ON_NODE:-20}

if [ -n "$SLURM_MEM_PER_NODE" ]; then
    MEM_GB=$((SLURM_MEM_PER_NODE / 1024))
else
    MEM_GB=500
fi

echo "[INFO] Using localcores = $CORES"
echo "[INFO] Using localmem   = $MEM_GB GB"

export PATH="/ictstr01/home/icb/raphael.kfuri-rubens/tools/xeniumranger-xenium4.0:$PATH"

xeniumranger resegment \
  --id="resegment_${SLIDE_NAME}" \
  --xenium-bundle="$SLIDE_DIR" \
  --localcores="$CORES" \
  --localmem="$MEM_GB"
