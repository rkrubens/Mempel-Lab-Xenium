import os
import pickle

import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    adata_path = "/home/icb/raphael.kfuri-rubens/git/mempel_xenium/resegmentation_adatas_merged/merged_MCA.h5ad"
    emb_dir = "/home/icb/raphael.kfuri-rubens/git/mempel_xenium/resegmentation_integration_embs"
    fig_dir = "/home/icb/raphael.kfuri-rubens/git/mempel_xenium/figures"

    os.makedirs(fig_dir, exist_ok=True)

    print(f"[INFO] Loading AnnData from {adata_path}")
    adata_mca = sc.read_h5ad(adata_path)

    if "X_pca" not in adata_mca.obsm:
        raise KeyError("X_pca not found in .obsm; run PCA first.")
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

    embed_keys = ["unintegrated", "harmony", "scvi", "drvi"]
    label_prefix = "MCA"

    for key in embed_keys:
        if key not in adata_mca.obsm:
            print(f"[WARN] Embedding '{key}' not found in .obsm, skipping.")
            continue

        print(f"[INFO] Computing neighbors and UMAP for embedding '{key}'")
        neighbors_key = f"neighbors_{key}"
        umap_key = f"umap_{key}"

        sc.pp.neighbors(adata_mca, use_rep=key, key_added=neighbors_key)

        sc.tl.umap(adata_mca, neighbors_key=neighbors_key, key_added=umap_key)

        basis = umap_key

        print(f"[INFO] Plotting UMAP for '{key}'")
        fig, ax = plt.subplots(figsize=(8, 8))
        sc.pl.embedding(
            adata_mca,
            basis=basis,
            color="slide_id",
            size=20,
            ax=ax,
            show=False,
            title=f"{label_prefix} - {key}",
            frameon=False
        )

        png_path = os.path.join(fig_dir, f"umap_{label_prefix}_{key}.png")
        pdf_path = os.path.join(fig_dir, f"umap_{label_prefix}_{key}.pdf")

        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"[INFO] Saved {png_path} and {pdf_path}")

    print("[INFO] All done.")


if __name__ == "__main__":
    main()
