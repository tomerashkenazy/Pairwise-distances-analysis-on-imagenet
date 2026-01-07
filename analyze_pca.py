import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import logging
import os
import argparse
from torchvision import datasets, transforms
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import seaborn as sns

# Set up logging
os.makedirs("logs", exist_ok=True)
log_file = os.path.join("logs", "analyze_pca.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Create plots directory
os.makedirs("results_imagenet_stats/plots", exist_ok=True)

def load_pca_results(norm):
    """Load PCA transformed data, model, and metadata for a given norm."""
    logger.info(f"Loading PCA results for {norm}...")
    
    transformed_path = f"results_imagenet_stats/val_pca_transformed_{norm}.pt"
    model_path = f"results_imagenet_stats/val_pca_model_{norm}.pkl"
    metadata_path = f"results_imagenet_stats/val_pca_metadata_{norm}.json"
    
    X_pca = torch.load(transformed_path).numpy()
    
    with open(model_path, 'rb') as f:
        pca_model = pickle.load(f)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Loaded PCA results: {X_pca.shape}")
    return X_pca, pca_model, metadata

def load_labels(eval_dir="/mnt/data/datasets/imagenet/val/"):
    """Load ImageNet validation labels."""
    logger.info("Loading ImageNet labels...")
    
    dataset_eval = datasets.ImageFolder(
        root=eval_dir,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    )
    
    labels = np.array([s[1] for s in dataset_eval.samples], dtype=np.int32)
    logger.info(f"Loaded {len(labels)} labels from {len(np.unique(labels))} classes")
    
    return labels

def plot_scree(metadata, norm, output_dir="results_imagenet_stats/plots"):
    """Plot scree plot showing explained variance ratio."""
    logger.info(f"Creating scree plot for {norm}...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Individual variance
    n_components = len(metadata['explained_variance_ratio'])

    components = np.arange(1, min(11, n_components + 1))
    ax1.bar(components, metadata['explained_variance_ratio'][:10])
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title(f'PCA Scree Plot - {norm.upper()}')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative variance
    ax2.plot(np.arange(1, n_components + 1), metadata['cumulative_variance_ratio'], 'b-', linewidth=2)
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    ax2.axhline(y=0.99, color='orange', linestyle='--', label='99% variance')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title(f'Cumulative Variance - {norm.upper()}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"pca_{norm}_scree.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved scree plot to {output_path}")

def plot_2d_projection(X_pca, labels, norm, output_dir="results_imagenet_stats/plots", max_classes=50):
    """Plot 2D projection of first two principal components."""
    logger.info(f"Creating 2D projection for {norm}...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Sample classes to avoid overcrowding
    unique_classes = np.unique(labels)
    if len(unique_classes) > max_classes:
        selected_classes = np.random.choice(unique_classes, max_classes, replace=False)
        mask = np.isin(labels, selected_classes)
        X_plot = X_pca[mask, :2]
        labels_plot = labels[mask]
    else:
        X_plot = X_pca[:, :2]
        labels_plot = labels
    
    scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=labels_plot, 
                        cmap='tab20', alpha=0.6, s=1)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(
    f'PCA 2D Projection - {norm.upper()} (showing {len(np.unique(labels_plot))} classes)',
    fontsize=22,
    fontweight='bold',
    pad=15
)

    plt.colorbar(scatter, ax=ax, label='Class ID')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"pca_{norm}_2d_projection.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved 2D projection to {output_path}")

def plot_3d_projection(X_pca, labels, norm, output_dir="results_imagenet_stats/plots", max_classes=50):
    """Plot 3D projection of first three principal components."""
    logger.info(f"Creating 3D projection for {norm}...")
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Sample classes to avoid overcrowding
    unique_classes = np.unique(labels)
    if len(unique_classes) > max_classes:
        selected_classes = np.random.choice(unique_classes, max_classes, replace=False)
        mask = np.isin(labels, selected_classes)
        X_plot = X_pca[mask, :3]
        labels_plot = labels[mask]
    else:
        X_plot = X_pca[:, :3]
        labels_plot = labels
    
    scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], X_plot[:, 2],
                        c=labels_plot, cmap='tab20', alpha=0.6, s=1)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title(
    f'PCA 3D Projection - {norm.upper()} (showing {len(np.unique(labels_plot))} classes)',
    fontsize=22,
    fontweight='bold',
    pad=15
)

    plt.colorbar(scatter, ax=ax, label='Class ID', shrink=0.8)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"pca_{norm}_3d_projection.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved 3D projection to {output_path}")

def compute_clustering_metrics(X_pca, labels, norm, n_components_list=[2, 3, 5, 10, 20, 50, 100]):
    """Compute clustering quality metrics for different numbers of components."""
    logger.info(f"Computing clustering metrics for {norm}...")
    
    results = {}
    
    for n_comp in n_components_list:
        if n_comp > X_pca.shape[1]:
            continue
        
        X_subset = X_pca[:, :n_comp]
        
        logger.info(f"  Computing metrics for {n_comp} components...")
        
        # Subsample for faster computation if dataset is large
        if len(X_subset) > 10000:
            indices = np.random.choice(len(X_subset), 10000, replace=False)
            X_sample = X_subset[indices]
            labels_sample = labels[indices]
        else:
            X_sample = X_subset
            labels_sample = labels
        
        silhouette = silhouette_score(X_sample, labels_sample, metric='euclidean')
        davies_bouldin = davies_bouldin_score(X_sample, labels_sample)
        calinski_harabasz = calinski_harabasz_score(X_sample, labels_sample)
        
        results[n_comp] = {
            'silhouette': float(silhouette),
            'davies_bouldin': float(davies_bouldin),
            'calinski_harabasz': float(calinski_harabasz)
        }
        
        logger.info(f"    Silhouette: {silhouette:.4f}")
        logger.info(f"    Davies-Bouldin: {davies_bouldin:.4f}")
        logger.info(f"    Calinski-Harabasz: {calinski_harabasz:.2f}")
    
    return results

def plot_clustering_metrics(metrics, norm, output_dir="results_imagenet_stats/plots"):
    """Plot clustering metrics vs number of components."""
    logger.info(f"Plotting clustering metrics for {norm}...")
    
    n_components = sorted(metrics.keys())
    silhouette = [metrics[n]['silhouette'] for n in n_components]
    davies_bouldin = [metrics[n]['davies_bouldin'] for n in n_components]
    calinski_harabasz = [metrics[n]['calinski_harabasz'] for n in n_components]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(n_components, silhouette, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Components')
    axes[0].set_ylabel('Silhouette Score')
    axes[0].set_title(f'Silhouette Score - {norm.upper()}')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(n_components, davies_bouldin, 'o-', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Davies-Bouldin Index')
    axes[1].set_title(f'Davies-Bouldin Index - {norm.upper()} (lower is better)')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(n_components, calinski_harabasz, 'o-', linewidth=2, markersize=8, color='green')
    axes[2].set_xlabel('Number of Components')
    axes[2].set_ylabel('Calinski-Harabasz Score')
    axes[2].set_title(f'Calinski-Harabasz Score - {norm.upper()}')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"pca_{norm}_clustering_metrics.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved clustering metrics plot to {output_path}")

def analyze_class_separability(X_pca, labels, norm, n_components=10):
    """Analyze per-class variance and separability in PCA space."""
    logger.info(f"Analyzing class separability for {norm}...")
    
    X_subset = X_pca[:, :n_components]
    unique_classes = np.unique(labels)
    
    # Compute per-class statistics
    class_means = []
    class_stds = []
    class_sizes = []
    
    for cls in unique_classes:
        mask = labels == cls
        class_data = X_subset[mask]
        class_means.append(np.mean(class_data, axis=0))
        class_stds.append(np.std(class_data, axis=0))
        class_sizes.append(len(class_data))
    
    class_means = np.array(class_means)
    class_stds = np.array(class_stds)
    
    # Compute inter-class distances (centroid distances)
    from scipy.spatial.distance import pdist, squareform
    inter_class_distances = squareform(pdist(class_means, metric='euclidean'))
    
    # Compute average intra-class variance
    avg_intra_class_var = np.mean([np.mean(std**2) for std in class_stds])
    avg_inter_class_dist = np.mean(inter_class_distances[np.triu_indices_from(inter_class_distances, k=1)])
    
    separability_ratio = avg_inter_class_dist / np.sqrt(avg_intra_class_var)
    
    results = {
        'avg_intra_class_variance': float(avg_intra_class_var),
        'avg_inter_class_distance': float(avg_inter_class_dist),
        'separability_ratio': float(separability_ratio),
        'n_classes': len(unique_classes),
        'n_components_used': n_components
    }
    
    logger.info(f"  Average intra-class variance: {avg_intra_class_var:.4f}")
    logger.info(f"  Average inter-class distance: {avg_inter_class_dist:.4f}")
    logger.info(f"  Separability ratio: {separability_ratio:.4f}")
    
    return results

def analyze_pca_for_norm(norm, eval_dir="/mnt/data/datasets/imagenet/val/"):
    """Run complete analysis for a given norm."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Analyzing PCA results for {norm}")
    logger.info(f"{'='*80}\n")
    
    # Load data
    X_pca, pca_model, metadata = load_pca_results(norm)
    labels = load_labels(eval_dir)
    
    # Ensure labels match transformed data
    assert len(labels) == len(X_pca), "Labels and PCA data length mismatch"
    
    # Generate plots
    plot_scree(metadata, norm)
    plot_2d_projection(X_pca, labels, norm)
    plot_3d_projection(X_pca, labels, norm)
    
    # Compute metrics
    clustering_metrics = compute_clustering_metrics(X_pca, labels, norm)
    plot_clustering_metrics(clustering_metrics, norm)
    
    separability = analyze_class_separability(X_pca, labels, norm, n_components=10)
    
    # Save all metrics
    analysis_results = {
        'norm': norm,
        'pca_metadata': metadata,
        'clustering_metrics': clustering_metrics,
        'class_separability': separability
    }
    
    output_path = f"results_imagenet_stats/pca_analysis_metrics_{norm}.json"
    with open(output_path, 'w') as f:
        json.dump(analysis_results, f, indent=4)
    logger.info(f"Saved analysis results to {output_path}")
    
    return analysis_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze PCA results')
    parser.add_argument('--norms', nargs='+', default=['l1', 'l2', 'linf'],
                      help='List of norms to analyze')
    parser.add_argument('--eval-dir', type=str, default="/mnt/data/datasets/imagenet/val/",
                      help='Path to ImageNet validation directory')
    
    args = parser.parse_args()
    
    all_results = {}
    for norm in args.norms:
        try:
            results = analyze_pca_for_norm(norm, args.eval_dir)
            all_results[norm] = results
        except Exception as e:
            logger.error(f"Error analyzing {norm}: {e}", exc_info=True)
            continue
    
    logger.info("\nAll norms analyzed successfully!")
