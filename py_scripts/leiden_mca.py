import scanpy as sc
import anndata as ad

in_path = "/home/icb/raphael.kfuri-rubens/git/mempel_xenium/resegmentation_final_adatas/mca_final.h5ad"
out_path = "/home/icb/raphael.kfuri-rubens/git/mempel_xenium/resegmentation_final_adatas/mca_leiden_final.h5ad"

print(f"[INFO] Reading AnnData: {in_path}")
adata = sc.read_h5ad(in_path)

resolutions = []
r = 0.1
while r <= 2.0001:
    resolutions.append(round(r, 3))
    r *= 2

print(f"[INFO] Computing Leiden for resolutions: {resolutions}")

for res in resolutions:
    key = f"leiden_{str(res).replace('.', '_')}"
    print(f"[INFO] Leiden resolution {res} -> {key}")
    sc.tl.leiden(
        adata,
        resolution=res,
        key_added=key,
        neighbors_key="neighbors_harmony"
    )

print(f"[INFO] Writing output: {out_path}")
adata.write(out_path)

print("[INFO] Done.")
