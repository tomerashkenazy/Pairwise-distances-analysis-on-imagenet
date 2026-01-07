# Pairwise Distances Analysis on ImageNet

This repository contains tools for analyzing pairwise distances between images in the ImageNet validation dataset. The project performs various statistical analyses including distance matrix computation, PCA dimensionality reduction, nearest neighbor classification, and visualization of class-level statistics.

## Overview

The main analyses performed in this project include:

- **Distance Matrix Computation**: Calculate pairwise distances between all images using L1, L2, and L-infinity norms
- **PCA Analysis**: Apply Principal Component Analysis to reduce dimensionality while preserving variance
- **Nearest Neighbor Classification**: Evaluate classification accuracy using nearest neighbor approaches
- **Class Statistics**: Compute per-class mean and standard deviation statistics
- **Distribution Analysis**: Analyze intra-class vs inter-class distance distributions using various metrics (Wasserstein distance, Jensen-Shannon divergence, ROC AUC)
- **Cluster Separation**: Evaluate separation quality using silhouette scores and other clustering metrics
- **Superclass Analysis**: Build and analyze WordNet-based superclass hierarchies

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for large-scale computations)
- ImageNet validation dataset

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tomerashkenazy/Pairwise-distances-analysis-on-imagenet.git
cd Pairwise-distances-analysis-on-imagenet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The project uses Hydra for configuration management. Edit `config.yaml` to set:

- `data_path`: Path to ImageNet validation dataset
- `bandwidth`: Bandwidth parameter for KDE
- `kernel`: Kernel type for density estimation
- `num_samples`: Number of samples to process
- `batch_size`: Batch size for data loading
- `num_workers`: Number of data loader workers
- `output_dir`: Directory for output files

## Usage

### 1. Compute Distance Matrices

Build pairwise distance matrices for the validation set:

```bash
python val_dist_mat.py
```

This will compute distance matrices using L1, L2, and L-infinity norms and save them to `results_imagenet_stats/`.

### 2. Apply PCA Dimensionality Reduction

Reduce dimensionality using incremental PCA:

```bash
python pca_distance_matrix.py --norms l1 l2 linf --variance 0.95 --batch-size 1000
```

Arguments:
- `--norms`: List of norms to process (default: l1, l2, linf)
- `--variance`: Variance threshold to retain (default: 0.95)
- `--batch-size`: Batch size for incremental PCA (default: 1000)

### 3. Nearest Neighbor Classification

Evaluate classification accuracy using nearest neighbors:

```bash
python nearest_neighbor.py
```

This script computes both standard and superclass-based nearest neighbor accuracies.

### 4. Compute Class Statistics

Calculate per-class mean and standard deviation:

```bash
python imagenet_class_stats.py
```

Results are saved to the directory specified in `config.yaml` (default: `class_stats/`).

### 5. Visualize Class Statistics

Generate visualizations for class-level statistics:

```bash
python visualize_class_stats.py
```

### 6. Distribution Analysis

Analyze intra-class vs inter-class distance distributions:

```bash
python imagenet_hist.py
```

This computes various metrics including:
- Wasserstein distance
- Jensen-Shannon divergence
- ROC AUC
- Mann-Whitney U statistic
- Bayes overlap

### 7. Cluster Separation Analysis

Evaluate cluster separation quality:

```bash
python cluster_seperation.py
```

Computes silhouette scores, Davies-Bouldin scores, and Calinski-Harabasz scores.

### 8. Build Superclass Matrix

Create WordNet-based superclass equivalence matrix:

```bash
python build_sup_mat.py
```

Note: Requires ImageNet devkit with meta.mat and wordnet.is_a.txt files.

## Project Structure

```
.
├── config.yaml                      # Hydra configuration file
├── val_dist_mat.py                  # Build distance matrices
├── pca_distance_matrix.py           # Apply PCA to distance matrices
├── nearest_neighbor.py              # NN classification evaluation
├── imagenet_class_stats.py          # Compute per-class statistics
├── visualize_class_stats.py         # Visualize class statistics
├── imagenet_hist.py                 # Distribution analysis
├── cluster_seperation.py            # Cluster quality metrics
├── plot_cluster_seperation.py       # Plot cluster separation results
├── build_sup_mat.py                 # Build superclass matrix
├── analyze_pca.py                   # PCA analysis utilities
├── compare_norms_pca.py             # Compare PCA across norms
├── view_mat.py                      # Matrix visualization
├── nn_display.py                    # Display NN results
├── kde_per_class.py                 # KDE per class
├── imagenet_kde_sklearn.py          # KDE using sklearn
├── imagenet_labels.txt              # ImageNet class labels
├── imagenet_depth7_equivalence.npy  # Superclass equivalence matrix
├── logs/                            # Log files
├── outputs/                         # Hydra outputs
├── class_stats/                     # Per-class statistics
└── results_imagenet_stats/          # Analysis results
```

## Output Files

Analysis results are saved in `results_imagenet_stats/`:

- `val_dist_matrix_{norm}.pt`: Distance matrices for each norm
- `val_pca_transformed_{norm}.pt`: PCA-transformed data
- `val_pca_model_{norm}.pkl`: Trained PCA models
- `val_pca_metadata_{norm}.json`: PCA metadata (variance explained, etc.)
- `imagenet_hist_geometry_{norm}.npy`: Histogram geometry data
- `imagenet_geometry_hist_metrics_{norm}.npy`: Distribution metrics
- `nn_results.json`: Nearest neighbor classification results

## Notes

- Large-scale distance matrix computations require substantial memory (50,000 × 50,000 matrices)
- GPU acceleration is strongly recommended for distance computations
- Some scripts require the ImageNet devkit for WordNet hierarchy information
- Log files are automatically saved to the `logs/` directory

## Citation

If you use this code in your research, please cite appropriately.

## License

See repository for license information.
