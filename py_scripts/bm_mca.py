import os
import pickle

import scanpy as sc
import pandas as pd

from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection


def main():
    adata_path = "/home/icb/raphael.kfuri-rubens/git/mempel_xenium/resegmentation_adatas_merged/merged_MCA.h5ad"
    emb_dir = "/home/icb/raphael.kfuri-rubens/git/mempel_xenium/resegmentation_integration_embs"
    outdir = "/home/icb/raphael.kfuri-rubens/git/mempel_xenium/integration_bm_results"

    os.makedirs(outdir, exist_ok=True)

    print(f"[INFO] Loading AnnData from {adata_path}")
    adata_mca = sc.read_h5ad(adata_path)

    adata_mca.obsm["unintegrated"] = adata_mca.obsm["X_pca"]

    print(f"[INFO] Loading embeddings from {emb_dir}")
    mca_files = [f for f in os.listdir(emb_dir) if "MCA" in f]

    for file in mca_files:
        filepath = os.path.join(emb_dir, file)
        print(f"[INFO] Reading embedding from {filepath}")
        with open(filepath, "rb") as f:
            emb = pickle.load(f)

        fname_lower = file.lower()
        if "drvi" in fname_lower:
            adata_mca.obsm["drvi"] = emb
        if "harmony" in fname_lower:
            adata_mca.obsm["harmony"] = emb
        if "scvi" in fname_lower:
            adata_mca.obsm["scvi"] = emb

    adata_mca.obs["cell_type"] = "cell"

    print("[INFO] Initializing Benchmarker")
    bm = Benchmarker(
        adata_mca,
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

    unscaled_path = os.path.join(outdir, "bm_df_unscaled_MCA.csv")
    scaled_path = os.path.join(outdir, "bm_df_scaled_MCA.csv")
    bm_path = os.path.join(outdir, "bm_MCA.pkl")

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
