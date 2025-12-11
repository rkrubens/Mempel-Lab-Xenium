import os
import sys
import traceback
from datetime import datetime

import spatialdata as sd
from spatialdata_io import xenium

import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import squidpy as sq


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main(base_dir: str) -> None:
    log("===== Xenium → AnnData resegmentation pipeline started =====")

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "N/A")
    slurm_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "N/A")
    log(f"SLURM_JOB_ID={slurm_job_id}, SLURM_ARRAY_TASK_ID={slurm_array_task_id}")
    log(f"Base directory: {base_dir}")

    if not os.path.isdir(base_dir):
        log(f"❌ Base directory does not exist or is not a directory: {base_dir}")
        sys.exit(1)

    zarr_out_dir = os.path.join(base_dir, "resegmentation_adatas")
    os.makedirs(zarr_out_dir, exist_ok=True)
    log(f"Output directory (h5ad): {zarr_out_dir}")

    bundle_folders = [
        f for f in os.listdir(base_dir)
        if f.startswith("resegment_output") and os.path.isdir(os.path.join(base_dir, f))
    ]

    if not bundle_folders:
        log("⚠️ No Xenium bundle folders found (starting with 'resegment_output'). Exiting.")
        return

    log(f"Found {len(bundle_folders)} bundles: {bundle_folders}")

    for i, folder in enumerate(bundle_folders, start=1):
        log("\n==============================================")
        log(f"[{i}/{len(bundle_folders)}] Processing Xenium bundle: {folder}")

        xenium_path = os.path.join(base_dir, folder, "outs")
        if not os.path.exists(xenium_path):
            log(f"❌ No 'outs/' folder found for {folder}, skipping.")
            continue

        adata_path = os.path.join(zarr_out_dir, f"{folder}.h5ad")

        try:
            log(f"→ Reading Xenium data from: {xenium_path}")
            sdata = xenium(
                xenium_path,
                cells_as_circles=True,
                morphology_mip=False,
                morphology_focus=False,
                aligned_images=False
            )

            if "table" not in sdata.tables:
                log(f"❌ No 'table' key found in sdata.tables for {folder}, skipping.")
                continue

            adata = sdata.tables["table"]
            log(f"   Loaded AnnData with shape: {adata.shape}")

            log("→ Calculating QC metrics...")
            sc.pp.calculate_qc_metrics(
                adata,
                percent_top=(10, 20, 50, 150),
                inplace=True,
            )

            n_cells_before = adata.n_obs
            n_genes_before = adata.n_vars

            log("→ Filtering cells (min_counts=10) and genes (min_cells=5)...")
            sc.pp.filter_cells(adata, min_counts=10)
            sc.pp.filter_genes(adata, min_cells=5)

            n_cells_after = adata.n_obs
            n_genes_after = adata.n_vars

            log(
                f"   Cells: {n_cells_before} → {n_cells_after}, "
                f"Genes: {n_genes_before} → {n_genes_after}"
            )

            log("→ Storing raw counts in adata.layers['counts']")
            adata.layers["counts"] = adata.X.copy()

            log(f"→ Saving AnnData to: {adata_path}")
            adata.write_h5ad(adata_path)
            log(f"✔️ Successfully saved: {adata_path}")

        except Exception as e:
            log(f"❌ Failed processing {folder}: {e}")
            log("Stack trace:")
            traceback.print_exc()
            continue

    log("\n===== All done. =====")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_dir_arg = sys.argv[1]
    else:
        base_dir_arg = "/home/icb/raphael.kfuri-rubens/git/mempel_xenium"

    main(base_dir_arg)
