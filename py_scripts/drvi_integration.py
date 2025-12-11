#!/usr/bin/env python3

import os
import sys
import pickle
import warnings
from datetime import datetime

import scanpy as sc
from drvi.model import DRVI
from rich import print

warnings.filterwarnings("ignore")


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main(
    input_path: str,
    layer: str = "counts",
    batch_key: str = "slide_id",
    feature_selection_key: str | None = None,
    n_latent: int = 20,
    encoder_dims: str = "256,128",
    decoder_dims: str = "128,256",
    n_epochs: int = 400,
    is_count_data: bool = True,
    early_stopping: bool = True,
    early_stopping_patience: int = 20,
    output_path: str | None = None,
):

    if not os.path.isfile(input_path):
        log(f"[bold red]❌ Input file not found:[/bold red] {input_path}")
        sys.exit(1)

    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}_drvi_embedding.pkl"

    def parse_dims(s):
        if isinstance(s, list):
            return s
        return [int(x) for x in s.split(",")]

    encoder_dims = parse_dims(encoder_dims)
    decoder_dims = parse_dims(decoder_dims)

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
            log(
                f"[bold blue]Subsetting to {mask.sum()} genes using "
                f"adata.var['{feature_selection_key}']==True.[/bold blue]"
            )
            adata = adata[:, mask].copy()
            log(f"   New AnnData shape after feature selection: {adata.shape}")

    if batch_key not in adata.obs:
        log(f"[bold red]❌ batch_key '{batch_key}' not in adata.obs[/bold red]")
        sys.exit(1)

    if layer is not None and layer not in adata.layers:
        log(
            f"[bold yellow]⚠️ layer '{layer}' not found in adata.layers; "
            "DRVI will use adata.X instead.[/bold yellow]"
        )
        layer_to_use = None
    else:
        layer_to_use = layer
        log(f"[bold blue]Using layer:[/bold blue] {layer_to_use}")

    log("[bold blue]Setting up DRVI AnnData...[/bold blue]")
    DRVI.setup_anndata(
        adata,
        layer=layer_to_use,
        categorical_covariate_keys=[batch_key],
        is_count_data=is_count_data,
    )

    log(
        f"[bold blue]Initializing DRVI: n_latent={n_latent}, "
        f"encoder_dims={encoder_dims}, decoder_dims={decoder_dims}[/bold blue]"
    )
    model = DRVI(
        adata,
        categorical_covariates=[batch_key],
        n_latent=n_latent,
        encoder_dims=encoder_dims,
        decoder_dims=decoder_dims,
    )

    log(
        f"[bold blue]Training DRVI for {n_epochs} epochs "
        f"(early_stopping={early_stopping}, patience={early_stopping_patience})...[/bold blue]"
    )
    model.train(
        max_epochs=n_epochs,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        plan_kwargs={"n_epochs_kl_warmup": n_epochs},
    )

    log("[bold blue]Extracting latent embedding...[/bold blue]")
    embedding = model.get_latent_representation()
    log(f"   Embedding shape: {embedding.shape}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    log(f"[bold green]Saving embedding to[/bold green] {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(embedding, f)

    log("[bold green]Process completed successfully![/bold green]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train DRVI and extract its latent embedding."
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--layer", type=str, default="counts")
    parser.add_argument("--batch-key", type=str, default="slide_id")
    parser.add_argument("--feature-selection-key", type=str, default=None)
    parser.add_argument("--n-latent", type=int, default=20)
    parser.add_argument("--encoder-dims", type=str, default="256,128")
    parser.add_argument("--decoder-dims", type=str, default="128,256")
    parser.add_argument("--n-epochs", type=int, default=400)
    parser.add_argument("--is-count-data", action="store_true", default=True)
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    main(
        input_path=args.input,
        layer=None if args.layer.lower() == "none" else args.layer,
        batch_key=args.batch_key,
        feature_selection_key=args.feature_selection_key,
        n_latent=args.n_latent,
        encoder_dims=args.encoder_dims,
        decoder_dims=args.decoder_dims,
        n_epochs=args.n_epochs,
        is_count_data=args.is_count_data,
        early_stopping=not args.no_early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        output_path=args.output,
    )
