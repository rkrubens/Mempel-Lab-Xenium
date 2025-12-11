#!/bin/bash -l
#SBATCH -o "outs/xenium_resegment_%j.out"
#SBATCH -e "outs/xenium_resegment_%j.err"
#SBATCH -J xenium_resegment_single
#SBATCH -q cpu_normal
#SBATCH -p cpu_p
#SBATCH -c 20
#SBATCH --mem=500GB
#SBATCH -t 24:00:00

XENIUM_BUNDLE="/ictstr01/groups/ml01/projects/2025_xenium_mempel_lab_raphael.kfuri-rubens/MCA205/resegmentation_data/output-XETG00212__0023280__Region_1__20240917__175222_MCA205_d21_"

SLIDE_NAME="$(basename "$XENIUM_BUNDLE")"

echo "[INFO] Using Xenium bundle: $XENIUM_BUNDLE"
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
    --xenium-bundle="$XENIUM_BUNDLE" \
    --localcores="$CORES" \
    --localmem="$MEM_GB"
