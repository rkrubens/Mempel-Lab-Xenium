import os
import sys
import pickle
import warnings
from datetime import datetime

from rich import print
import scanpy as sc

warnings.filterwarnings("ignore")


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main(
    input_path: str,
    batch_key: str = "slide_id",
    basis: str = "X_pca",
    adjusted_basis: str = "X_pca_harmony",
    output_path: str | None = None,
) -> None:

    if not os.path.isfile(input_path):
        log(f"[bold red]❌ Input file not found:[/bold red] {input_path}")
        sys.exit(1)

    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}_harmony_embedding.pkl"

    log(f"[bold green]Loading AnnData from[/bold green] {input_path}")
    adata = sc.read_h5ad(input_path)
    log(f"   AnnData shape: {adata.shape}")

    if batch_key not in adata.obs:
        log(f"[bold red]❌ batch_key '{batch_key}' not found in adata.obs[/bold red]")
        sys.exit(1)
    else:
        log(f"[bold blue]Using batch_key:[/bold blue] '{batch_key}'")

    if basis not in adata.obsm:
        if basis == "X_pca":
            log(f"[bold yellow]{basis} not found, computing PCA...[/bold yellow]")

            mask_var = None
            if "highly_variable" in adata.var:
                log("   'highly_variable' found in adata.var, using HVGs for PCA.")
                mask_var = adata.var["highly_variable"]

            if not (adata.X.dtype == "float32" or adata.X.dtype == "float64"):
                log("   Casting adata.X to float32 for PCA.")
                adata.X = adata.X.astype("float32")

            sc.tl.pca(adata, use_highly_variable=mask_var is not None)
            log(f"   PCA computed, basis stored in adata.obsm['{basis}']")
        else:
            log(
                f"[bold red]Error: basis '{basis}' not found in adata.obsm "
                f"and only automatic support for 'X_pca' is implemented.[/bold red]"
            )
            sys.exit(1)
    else:
        log(f"[bold blue]Using precomputed basis:[/bold blue] '{basis}'")

    log("[bold blue]Applying Harmony integration...[/bold blue]")
    sc.external.pp.harmony_integrate(
        adata,
        key=batch_key,
        basis=basis,
        adjusted_basis=adjusted_basis,
    )

    if adjusted_basis not in adata.obsm:
        log(
            f"[bold red]❌ Harmony did not create '{adjusted_basis}' in adata.obsm.[/bold red]"
        )
        sys.exit(1)

    log("[bold blue]Extracting Harmony embeddings...[/bold blue]")
    embeddings = adata.obsm[adjusted_basis].copy()
    log(f"   Embedding shape: {embeddings.shape}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    log(f"[bold green]Saving embeddings to[/bold green] {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(embeddings, f)

    log("[bold green]Process completed successfully![/bold green]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Harmony on an AnnData file and save the resulting embedding."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input .h5ad file (e.g. merged_TU.h5ad).",
    )
    parser.add_argument(
        "--batch-key",
        type=str,
        default="slide_id",
        help="Column in adata.obs to use as batch key (default: slide_id).",
    )
    parser.add_argument(
        "--basis",
        type=str,
        default="X_pca",
        help="Embedding in adata.obsm to use as basis (default: X_pca).",
    )
    parser.add_argument(
        "--adjusted-basis",
        type=str,
        default="X_pca_harmony",
        help="Name for Harmony-corrected embedding in adata.obsm (default: X_pca_harmony).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Path to output .pkl file for embeddings. "
            "If not given, uses <input_basename>_harmony_embedding.pkl"
        ),
    )

    args = parser.parse_args()

    main(
        input_path=args.input,
        batch_key=args.batch_key,
        basis=args.basis,
        adjusted_basis=args.adjusted_basis,
        output_path=args.output,
    )
