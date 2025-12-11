# Xenium Resegmentation and Integration Benchmarking

This repository contains workflows and scripts developed for the **Mempel Lab** (CIID at Massachusetts General Hospital, Harvard Medical School) to resegment **YUMM** and **MCA205** Xenium samples, perform quality control, integrate samples across tumors, and benchmark multiple integration methods for downstream annotation.

## Overview

The pipeline processes Xenium spatial transcriptomics data through the following stages:

### 1. Resegmentation

Resegmentation was performed using:

* **Xenium Ranger 4.0.0** (Sep 2, 2025)
  10x Genomics: [https://www.10xgenomics.com/support/software/xenium-ranger/downloads](https://www.10xgenomics.com/support/software/xenium-ranger/downloads)

* **Proseg v3.0.11**, Jones *et al.*, *Nature Methods*, 2025
  Repository: [https://github.com/dcjones/proseg](https://github.com/dcjones/proseg)

### 2. Slide Ingestion and Quality Control

Following resegmentation:

* Individual Xenium slides were ingested
* Dataset-level **quality control** was performed
* All slides were **concatenated** into a unified AnnData object per tumor model

### 3. Integration Benchmark per Tumor

Each tumor model (YUMM, MCA205) underwent an integration benchmarking experiment using **three methods**, implemented via the **scIB** framework:

* **Harmony**
  Korsunsky *et al.*, *Nature Methods*, 2019

* **scVI**
  Lopez *et al.*, *Nature Methods*, 2018

* **drVI**
  Moinfar *et al.*, bioRxiv, 2025

Integration was evaluated following the scIB benchmarking guidelines (Luecken *et al.*, *Nature Methods*, 2022).

### 4. Graph Construction and Clustering

For each integration result:

* **kNN graphs** were computed based on the corresponding embedding
* **UMAP** embeddings were generated for visualization
* **Leiden clustering** was performed at resolutions **0.1, 0.2, 0.4, …, 2.0**, producing a series of cluster labels (e.g., `leiden_0_1`, `leiden_0_2`, …)

---

## Citation

* Jones *et al.*, **Proseg**, *Nature Methods*, 2025
* Korsunsky *et al.*, **Harmony**, *Nature Methods*, 2019
* Lopez *et al.*, **scVI**, *Nature Methods*, 2018
* Moinfar *et al.*, **drVI**, bioRxiv, 2025
* Luecken *et al.*, **scIB**, *Nature Methods*, 2022
* 10x Genomics **Xenium Ranger** documentation
