import os
import pickle

import scanpy as sc
import pandas as pd

from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection


def main():
    adata_path = "/home/icb/raphael.kfuri-rubens/git/mempel_xenium/resegmentation_adatas_merged/merged_TU.h5ad"
    emb_dir = "/home/icb/raphael.kfuri-rubens/git/mempel_xenium/resegmentation_integration_embs"
    outdir = "/home/icb/raphael.kfuri-rubens/git/mempel_xenium/integration_bm_results"

    os.makedirs(outdir, exist_ok=True)

    print(f"[INFO] Loading AnnData from {adata_path}")
    adata_tu = sc.read_h5ad(adata_path)

    adata_tu.obsm["unintegrated"] = adata_tu.obsm["X_pca"]

    print(f"[INFO] Loading embeddings from {emb_dir}")
    tu_files = [f for f in os.listdir(emb_dir) if "TU" in f]

    for file in tu_files:
        filepath = os.path.join(emb_dir, file)
        print(f"[INFO] Reading embedding from {filepath}")
        with open(filepath, "rb") as f:
            emb = pickle.load(f)

        fname_lower = file.lower()
        if "drvi" in fname_lower:
            adata_tu.obsm["drvi"] = emb
        if "harmony" in fname_lower:
            adata_tu.obsm["harmony"] = emb
        if "scvi" in fname_lower:
            adata_tu.obsm["scvi"] = emb

    adata_tu.obs["cell_type"] = "cell"

    print("[INFO] Initializing Benchmarker")
    bm = Benchmarker(
        adata_tu,
        batch_key="slide_id",
        label_key="cell_type",
        bio_conservation_metrics=None,
        batch_correction_metrics=BatchCorrection(),
        embedding_obsm_keys=["unintegrated", "harmony", "scvi", "drvi"],
        n_jobs=20,
    )

    print("[INFO] Running benchmark")
    bm.benchmark()

    print("[INFO] Collecting results")
    df_non_scaled = bm.get_results(min_max_scale=False)
    df_scaled = bm.get_results(min_max_scale=True)

    unscaled_path = os.path.join(outdir, "bm_df_unscaled.csv")
    scaled_path = os.path.join(outdir, "bm_df_scaled.csv")
    bm_path = os.path.join(outdir, "bm.pkl")

    print(f"[INFO] Saving unscaled results to {unscaled_path}")
    df_non_scaled.to_csv(unscaled_path)

    print(f"[INFO] Saving scaled results to {scaled_path}")
    df_scaled.to_csv(scaled_path)

    print(f"[INFO] Saving Benchmarker object to {bm_path}")
    with open(bm_path, "wb") as f:
        pickle.dump(bm, f)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
