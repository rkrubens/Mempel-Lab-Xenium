#!/usr/bin/env python3

import os
import sys
from datetime import datetime
import traceback

import scanpy as sc


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    base_dir = "/home/icb/raphael.kfuri-rubens/git/mempel_xenium"
    in_dir = os.path.join(base_dir, "resegmentation_adatas")
    out_dir = os.path.join(base_dir, "resegmentation_adatas_merged")

    log("===== Start concatenating resegmented AnnData files =====")
    log(f"Input directory: {in_dir}")
    log(f"Output directory: {out_dir}")

    if not os.path.isdir(in_dir):
        log(f"❌ Input directory does not exist: {in_dir}")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    all_files = [
        f for f in os.listdir(in_dir)
        if f.endswith(".h5ad") and os.path.isfile(os.path.join(in_dir, f))
    ]
    log(f"Found {len(all_files)} .h5ad files: {all_files}")

    tu_files = [f for f in all_files if "_TU_" in f]
    mca_files = [f for f in all_files if "MCA" in f]

    log(f"TU files ({len(tu_files)}): {tu_files}")
    log(f"MCA files ({len(mca_files)}): {mca_files}")

    def load_and_concat(file_list, label: str):
        if not file_list:
            log(f"⚠️ No files for group '{label}', skipping concatenation.")
            return None

        adatas = []
        slide_ids = []

        for i, fname in enumerate(file_list, start=1):
            fpath = os.path.join(in_dir, fname)
            slide_id = os.path.splitext(fname)[0]
            log(f"[{label} {i}/{len(file_list)}] Reading: {fpath} (slide_id={slide_id})")

            try:
                ad = sc.read_h5ad(fpath)
            except Exception as e:
                log(f"❌ Failed to read {fpath}: {e}")
                traceback.print_exc()
                continue

            ad.obs["slide_id"] = slide_id
            adatas.append(ad)
            slide_ids.append(slide_id)
            log(f"   Loaded AnnData with shape {ad.shape}")

        if not adatas:
            log(f"⚠️ No valid AnnData objects loaded for group '{label}'.")
            return None

        log(f"→ Concatenating {len(adatas)} AnnData objects for group '{label}'...")
        merged = sc.concat(
            adatas,
            join="outer",
            label=None,
            index_unique="slide_id-"
        )
        log(f"   Resulting merged AnnData shape: {merged.shape}")
        return merged

    tu_merged = load_and_concat(tu_files, label="TU")
    if tu_merged is not None:
        print("→ Preprocessing TU merged AnnData...")
        sc.pp.normalize_total(tu_merged, inplace=True)
        print("→ Log1p transformation...")
        sc.pp.log1p(tu_merged)
        print("→ PCA...")
        sc.pp.pca(tu_merged)
        print("→ Neighbors...")
        sc.pp.neighbors(tu_merged)
        print("→ UMAP...")
        sc.tl.umap(tu_merged)
        tu_out = os.path.join(out_dir, "merged_TU.h5ad")
        log(f"→ Saving TU merged AnnData to: {tu_out}")
        tu_merged.write_h5ad(tu_out)
        log(f"✔️ Saved TU merged file: {tu_out}")

    mca_merged = load_and_concat(mca_files, label="MCA")
    if mca_merged is not None:
        print("→ Preprocessing MCA merged AnnData...")
        sc.pp.normalize_total(mca_merged, inplace=True)
        print("→ Log1p transformation...")
        sc.pp.log1p(mca_merged)
        print("→ PCA...")
        sc.pp.pca(mca_merged)
        print("→ Neighbors...")
        sc.pp.neighbors(mca_merged)
        print("→ UMAP...")
        sc.tl.umap(mca_merged)
        mca_out = os.path.join(out_dir, "merged_MCA.h5ad")
        log(f"→ Saving MCA merged AnnData to: {mca_out}")
        mca_merged.write_h5ad(mca_out)
        log(f"✔️ Saved MCA merged file: {mca_out}")

    log("===== Done. =====")


if __name__ == "__main__":
    main()
