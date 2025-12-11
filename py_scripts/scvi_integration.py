import os
import sys
import pickle
from datetime import datetime
import warnings

import scanpy as sc
import scvi
from rich import print

warnings.filterwarnings("ignore")


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main(
    input_path: str,
    layer: str = "counts",
    batch_key: str = "slide_id",
    n_layers: int = 2,
    n_latent: int = 30,
    gene_likelihood: str = "nb",
    max_epochs: int = 400,
    feature_selection_key: str | None = None,
    output_path: str | None = None,
) -> None:

    if not os.path.isfile(input_path):
        log(f"[bold red]❌ Input file not found:[/bold red] {input_path}")
        sys.exit(1)

    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}_scvi_embedding.pkl"

    log(f"[bold green]Loading AnnData from[/bold green] {input_path}")
    adata = sc.read_h5ad(input_path)
    log(f"   AnnData shape: {adata.shape}")

    if feature_selection_key is not None:
        if feature_selection_key not in adata.var:
            log(
                f"[bold yellow]⚠️ feature_selection_key '{feature_selection_key}' "
                f"not found in adata.var; using all genes.[/bold yellow]"
            )
        else:
            mask = adata.var[feature_selection_key].astype(bool)
            n_selected = mask.sum()
            log(
                f"[bold blue]Subsetting to features where "
                f"adata.var['{feature_selection_key}'] == True "
                f"({n_selected} genes).[/bold blue]"
            )
            adata = adata[:, mask].copy()
            log(f"   New AnnData shape after feature selection: {adata.shape}")

    if batch_key not in adata.obs:
        log(f"[bold red]❌ batch_key '{batch_key}' not found in adata.obs[/bold red]")
        sys.exit(1)
    else:
        log(f"[bold blue]Using batch_key:[/bold blue] '{batch_key}'")

    if layer is not None and layer not in adata.layers:
        log(
            f"[bold yellow]⚠️ layer '{layer}' not found in adata.layers; "
            "using adata.X instead.[/bold yellow]"
        )
        layer_to_use = None
    else:
        layer_to_use = layer
        log(f"[bold blue]Using layer:[/bold blue] '{layer_to_use}'")

    scvi.settings.seed = 0
    log("[bold blue]Setting up scVI AnnData...[/bold blue]")
    scvi.model.SCVI.setup_anndata(
        adata,
        layer=layer_to_use,
        batch_key=batch_key,
    )

    log(
        f"[bold blue]Initializing scVI model "
        f"(n_layers={n_layers}, n_latent={n_latent}, gene_likelihood='{gene_likelihood}').[/bold blue]"
    )
    model = scvi.model.SCVI(
        adata,
        n_layers=n_layers,
        n_latent=n_latent,
        gene_likelihood=gene_likelihood,
    )

    log(f"[bold blue]Training scVI model for {max_epochs} epochs...[/bold blue]")
    model.train(max_epochs=max_epochs)

    log("[bold blue]Extracting scVI latent embedding...[/bold blue]")
    embedding = model.get_latent_representation()
    log(f"   Embedding shape: {embedding.shape}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    log(f"[bold green]Saving scVI embedding to[/bold green] {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(embedding, f)

    log("[bold green]Process completed successfully![/bold green]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train an scVI model on an AnnData file and extract the latent embedding."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input .h5ad file (e.g. merged_TU.h5ad).",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="counts",
        help="Layer to use as input (default: counts). Use None to use adata.X.",
    )
    parser.add_argument(
        "--batch-key",
        type=str,
        default="slide_id",
        help="Column in adata.obs used as batch key (default: slide_id).",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=2,
        help="Number of hidden layers in the encoder/decoder (default: 2).",
    )
    parser.add_argument(
        "--n-latent",
        type=int,
        default=30,
        help="Dimensionality of the latent space (default: 30).",
    )
    parser.add_argument(
        "--gene-likelihood",
        type=str,
        default="nb",
        choices=["nb", "zinb", "poisson"],
        help="Gene likelihood for scVI (default: nb).",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=400,
        help="Maximum number of training epochs (default: 400).",
    )
    parser.add_argument(
        "--feature-selection-key",
        type=str,
        default=None,
        help=(
            "Optional adata.var boolean column used to subset genes before scVI "
            "(e.g. 'highly_variable'). If None, use all genes."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Path to output .pkl file for latent embedding. "
            "If not given, uses <input_basename>_scvi_embedding.pkl"
        ),
    )

    args = parser.parse_args()

    main(
        input_path=args.input,
        layer=None if args.layer.lower() == "none" else args.layer,
        batch_key=args.batch_key,
        n_layers=args.n_layers,
        n_latent=args.n_latent,
        gene_likelihood=args.gene_likelihood,
        max_epochs=args.max_epochs,
        feature_selection_key=args.feature_selection_key,
        output_path=args.output,
    )
