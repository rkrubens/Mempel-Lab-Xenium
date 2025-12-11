#!/bin/bash -l
#SBATCH -o "outs/proseg_%A_%a.out"
#SBATCH -e "outs/proseg_%A_%a.err"
#SBATCH -J proseg_run
#SBATCH -q gpu_normal
#SBATCH -p gpu_p
#SBATCH -c 20
#SBATCH --mem=500GB
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH --array=0-11

RESEGMENT_ROOT="/ictstr01/home/icb/raphael.kfuri-rubens/git/mempel_xenium"

PROSEG_OUT="/ictstr01/home/icb/raphael.kfuri-rubens/git/mempel_xenium/proseg_outs"

mkdir -p "$PROSEG_OUT"

export PATH="/home/icb/raphael.kfuri-rubens/.cargo/bin:$PATH"

mapfile -t SLIDE_DIRS < <(find "$RESEGMENT_ROOT" -maxdepth 1 -mindepth 1 -type d -name "resegment_*" | sort)

TOTAL=${#SLIDE_DIRS[@]}

if [ "$TOTAL" -eq 0 ]; then
    echo "[ERROR] No resegment_* directories found in $RESEGMENT_ROOT"
    exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL" ]; then
    echo "[INFO] Task ID $SLURM_ARRAY_TASK_ID exceeds number of slides ($TOTAL). Exiting."
    exit 0
fi

RESEG_DIR="${SLIDE_DIRS[$SLURM_ARRAY_TASK_ID]}"
BASENAME="$(basename "$RESEG_DIR")"

SAMPLE_NAME="${BASENAME#resegment_}"

TRANSCRIPTS="$RESEG_DIR/outs/transcripts.parquet"
OUTPUT_PATH="$PROSEG_OUT/${SAMPLE_NAME}.zarr"

echo "[INFO] Processing: $BASENAME"
echo "[INFO] transcripts = $TRANSCRIPTS"
echo "[INFO] output      = $OUTPUT_PATH"

if [ ! -f "$TRANSCRIPTS" ]; then
    echo "[ERROR] transcripts.parquet not found for $BASENAME"
    exit 1
fi

proseg \
  --xenium "$TRANSCRIPTS" \
  --output-spatialdata "$OUTPUT_PATH"
