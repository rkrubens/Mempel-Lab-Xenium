import os
import sys
import scanpy as sc


def main():
    file_cfgs = [
        {
            "input": "/home/icb/raphael.kfuri-rubens/git/mempel_xenium/resegmentation_final_adatas/mca_pre_final.h5ad",
            "output": "/home/icb/raphael.kfuri-rubens/git/mempel_xenium/resegmentation_final_adatas/mca_final.h5ad",
        },
        {
            "input": "/home/icb/raphael.kfuri-rubens/git/mempel_xenium/resegmentation_final_adatas/tu_pre_final.h5ad",
            "output": "/home/icb/raphael.kfuri-rubens/git/mempel_xenium/resegmentation_final_adatas/tu_final.h5ad",
        },
    ]

    task_id_str = os.environ.get("SLURM_ARRAY_TASK_ID", None)
    if task_id_str is None:
        print("[ERROR] SLURM_ARRAY_TASK_ID not set. Are you running inside an array job?")
        sys.exit(1)

    try:
        task_id = int(task_id_str)
    except ValueError:
        print(f"[ERROR] Invalid SLURM_ARRAY_TASK_ID: {task_id_str}")
        sys.exit(1)

    if task_id < 0 or task_id >= len(file_cfgs):
        print(f"[ERROR] SLURM_ARRAY_TASK_ID={task_id} out of range for file_cfgs (len={len(file_cfgs)})")
        sys.exit(1)

    cfg = file_cfgs[task_id]
    input_path = cfg["input"]
    output_path = cfg["output"]

    print(f"[INFO] SLURM_ARRAY_TASK_ID = {task_id}")
    print(f"[INFO] Loading AnnData from: {input_path}")
    adata = sc.read_h5ad(input_path)

    embeddings = ["unintegrated", "drvi", "scvi", "harmony"]

    for rep in embeddings:
        if rep not in adata.obsm_keys():
            print(f"[WARNING] Embedding '{rep}' not found in adata.obsm. Skipping.")
            continue

        neighbors_key = f"neighbors_{rep}"
        umap_key = f"X_umap_{rep}"

        if neighbors_key in adata.uns:
            print(f"[INFO] Neighbors already computed for '{rep}' (key: '{neighbors_key}'). Skipping neighbors.")
        else:
            print(f"[INFO] Computing neighbors for embedding '{rep}' with key_added='{neighbors_key}'...")
            sc.pp.neighbors(
                adata,
                use_rep=rep,
                key_added=neighbors_key,
            )

        if umap_key in adata.obsm_keys():
            print(f"[INFO] UMAP already present for '{rep}' (obsm['{umap_key}']). Skipping UMAP.")
        else:
            print(f"[INFO] Computing UMAP for embedding '{rep}' with neighbors_key='{neighbors_key}', key_added='{umap_key}'...")
            sc.tl.umap(
                adata,
                neighbors_key=neighbors_key,
                key_added=umap_key,
                min_dist=0.1,
            )

    print(f"[INFO] Writing updated AnnData to: {output_path}")
    adata.write_h5ad(output_path)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
