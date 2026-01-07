import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import logging
import os
import argparse
from scipy.stats import pearsonr, spearmanr
import seaborn as sns

# Set up logging
os.makedirs("logs", exist_ok=True)
log_file = os.path.join("logs", "compare_norms_pca.log")
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

def load_all_results(norms):
    """Load PCA results and analysis for all norms."""
    logger.info("Loading results for all norms...")
    
    results = {}
    for norm in norms:
        try:
            # Load transformed data
            transformed_path = f"results_imagenet_stats/val_pca_transformed_{norm}.pt"
            X_pca = torch.load(transformed_path).numpy()
            
            # Load analysis results
            analysis_path = f"results_imagenet_stats/pca_analysis_metrics_{norm}.json"
            with open(analysis_path, 'r') as f:
                analysis = json.load(f)
            
            results[norm] = {
                'X_pca': X_pca,
                'analysis': analysis
            }
            logger.info(f"  Loaded {norm}: {X_pca.shape}")
        except Exception as e:
            logger.error(f"  Error loading {norm}: {e}")
            continue
    
    return results

def plot_scree_comparison(results, output_dir="results_imagenet_stats/plots"):
    """Compare scree plots across all norms."""
    logger.info("Creating scree plot comparison...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = {'l1': 'blue', 'l2': 'red', 'linf': 'green'}
    
    # Individual variance (first 50 components)
    for norm, data in results.items():
        var_ratios = data['analysis']['pca_metadata']['explained_variance_ratio'][:50]
        components = np.arange(1, len(var_ratios) + 1)
        axes[0].plot(components, var_ratios, 'o-', label=norm.upper(), 
                    color=colors.get(norm, 'black'), alpha=0.7, linewidth=2)
    
    axes[0].set_xlabel('Principal Component', fontsize=12)
    axes[0].set_ylabel('Explained Variance Ratio', fontsize=12)
    axes[0].set_title('PCA Scree Plot - All Norms', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Cumulative variance
    for norm, data in results.items():
        cumsum = data['analysis']['pca_metadata']['cumulative_variance_ratio']
        components = np.arange(1, len(cumsum) + 1)
        axes[1].plot(components, cumsum, '-', label=norm.upper(), 
                    color=colors.get(norm, 'black'), alpha=0.7, linewidth=2)
    
    axes[1].axhline(y=0.95, color='gray', linestyle='--', label='95% variance', alpha=0.5)
    axes[1].axhline(y=0.99, color='gray', linestyle=':', label='99% variance', alpha=0.5)
    axes[1].set_xlabel('Number of Components', fontsize=12)
    axes[1].set_ylabel('Cumulative Explained Variance', fontsize=12)
    axes[1].set_title('Cumulative Variance - All Norms', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "pca_norm_comparison_scree.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved scree comparison to {output_path}")

def plot_2d_projections_comparison(results, labels, output_dir="results_imagenet_stats/plots", n_samples=5000):
    """Create side-by-side 2D projections for all norms."""
    logger.info("Creating 2D projection comparison...")
    
    norms = list(results.keys())
    n_norms = len(norms)
    
    fig, axes = plt.subplots(1, n_norms, figsize=(6*n_norms, 5))
    if n_norms == 1:
        axes = [axes]
    
    # Sample data for visualization
    indices = np.random.choice(len(labels), min(n_samples, len(labels)), replace=False)
    labels_sample = labels[indices]
    
    for idx, norm in enumerate(norms):
        X_pca = results[norm]['X_pca']
        X_sample = X_pca[indices, :2]
        
        scatter = axes[idx].scatter(X_sample[:, 0], X_sample[:, 1], 
                                   c=labels_sample, cmap='tab20', 
                                   alpha=0.5, s=1)
        axes[idx].set_xlabel('PC1', fontsize=11)
        axes[idx].set_ylabel('PC2', fontsize=11)
        axes[idx].set_title(f'{norm.upper()}', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "pca_norm_comparison_2d_projections.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved 2D projection comparison to {output_path}")

def plot_clustering_metrics_comparison(results, output_dir="results_imagenet_stats/plots"):
    """Compare clustering metrics across norms."""
    logger.info("Creating clustering metrics comparison...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {'l1': 'blue', 'l2': 'red', 'linf': 'green'}
    
    for norm, data in results.items():
        metrics = data['analysis']['clustering_metrics']
        n_components = sorted([int(k) for k in metrics.keys()])
        
        silhouette = [metrics[str(n)]['silhouette'] for n in n_components]
        davies_bouldin = [metrics[str(n)]['davies_bouldin'] for n in n_components]
        calinski_harabasz = [metrics[str(n)]['calinski_harabasz'] for n in n_components]
        
        axes[0].plot(n_components, silhouette, 'o-', label=norm.upper(), 
                    color=colors.get(norm, 'black'), linewidth=2, markersize=8, alpha=0.7)
        axes[1].plot(n_components, davies_bouldin, 'o-', label=norm.upper(), 
                    color=colors.get(norm, 'black'), linewidth=2, markersize=8, alpha=0.7)
        axes[2].plot(n_components, calinski_harabasz, 'o-', label=norm.upper(), 
                    color=colors.get(norm, 'black'), linewidth=2, markersize=8, alpha=0.7)
    
    axes[0].set_xlabel('Number of Components', fontsize=11)
    axes[0].set_ylabel('Silhouette Score', fontsize=11)
    axes[0].set_title('Silhouette Score Comparison', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Number of Components', fontsize=11)
    axes[1].set_ylabel('Davies-Bouldin Index', fontsize=11)
    axes[1].set_title('Davies-Bouldin Index (lower is better)', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Number of Components', fontsize=11)
    axes[2].set_ylabel('Calinski-Harabasz Score', fontsize=11)
    axes[2].set_title('Calinski-Harabasz Score Comparison', fontsize=12, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "pca_norm_comparison_clustering_metrics.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved clustering metrics comparison to {output_path}")

def compute_projection_correlations(results, n_components=10):
    """Compute correlations between PCA projections across norms."""
    logger.info("Computing projection correlations across norms...")
    
    norms = list(results.keys())
    
    # Determine safe number of components based on available data
    min_comps = min(results[n]['X_pca'].shape[1] for n in norms)
    if min_comps < n_components:
        logger.info(f"Limiting correlation analysis to {min_comps} components (limited by available dimensions).")
        n_components = min_comps

    correlations = {}
    
    for i, norm1 in enumerate(norms):
        for norm2 in norms[i+1:]:
            X1 = results[norm1]['X_pca'][:, :n_components]
            X2 = results[norm2]['X_pca'][:, :n_components]
            
            # Compute correlation for each component pair
            component_corrs = []
            for comp in range(n_components):
                corr, _ = pearsonr(X1[:, comp], X2[:, comp])
                component_corrs.append(abs(corr))  # Use absolute correlation
            
            avg_corr = np.mean(component_corrs)
            correlations[f"{norm1}_vs_{norm2}"] = {
                'component_correlations': component_corrs,
                'average_correlation': float(avg_corr)
            }
            
            logger.info(f"  {norm1} vs {norm2}: avg correlation = {avg_corr:.4f}")
    
    return correlations

def plot_projection_correlations(correlations, output_dir="results_imagenet_stats/plots"):
    """Plot correlation heatmap between norm projections."""
    logger.info("Creating projection correlation plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pairs = list(correlations.keys())
    n_components = len(correlations[pairs[0]]['component_correlations'])
    
    # Create matrix for heatmap
    corr_matrix = np.array([correlations[pair]['component_correlations'] 
                           for pair in pairs])
    
    im = ax.imshow(corr_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    ax.set_yticks(np.arange(len(pairs)))
    ax.set_yticklabels([p.replace('_', ' ').upper() for p in pairs])
    ax.set_xticks(np.arange(n_components))
    ax.set_xticklabels([f'PC{i+1}' for i in range(n_components)])
    ax.set_xlabel('Principal Component', fontsize=11)
    ax.set_title('Correlation between Norm Projections', fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', fontsize=11)
    
    # Add correlation values
    for i in range(len(pairs)):
        for j in range(n_components):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "pca_norm_comparison_correlations.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved correlation plot to {output_path}")

def create_summary_table(results, correlations):
    """Create summary comparison table."""
    logger.info("Creating summary comparison table...")
    
    summary = {}
    
    for norm, data in results.items():
        metadata = data['analysis']['pca_metadata']
        separability = data['analysis']['class_separability']
        
        # Get 10-component metrics
        metrics_10 = data['analysis']['clustering_metrics'].get('10', {})
        
        summary[norm] = {
            'n_components_95pct': metadata['n_components'],
            'first_pc_variance': metadata['explained_variance_ratio'][0],
            'top5_cumulative_variance': metadata['cumulative_variance_ratio'][4] if len(metadata['cumulative_variance_ratio']) > 4 else metadata['cumulative_variance_ratio'][-1],
            'silhouette_10comp': metrics_10.get('silhouette', None),
            'davies_bouldin_10comp': metrics_10.get('davies_bouldin', None),
            'calinski_harabasz_10comp': metrics_10.get('calinski_harabasz', None),
            'separability_ratio': separability['separability_ratio'],
            'avg_inter_class_dist': separability['avg_inter_class_distance'],
            'avg_intra_class_var': separability['avg_intra_class_variance']
        }
    
    # Add correlation summary
    correlation_summary = {}
    for pair, corr_data in correlations.items():
        correlation_summary[pair] = corr_data['average_correlation']
    
    summary['correlations'] = correlation_summary
    
    return summary

def compare_norms(norms, labels):
    """Main comparison function."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Comparing PCA results across norms: {norms}")
    logger.info(f"{'='*80}\n")
    
    # Load all results
    results = load_all_results(norms)
    
    if len(results) < 2:
        logger.error("Need at least 2 norms to compare!")
        return
    
    # Generate comparison plots
    plot_scree_comparison(results)
    plot_2d_projections_comparison(results, labels)
    plot_clustering_metrics_comparison(results)
    
    # Compute and plot correlations
    correlations = compute_projection_correlations(results, n_components=10)
    plot_projection_correlations(correlations)
    
    # Create summary
    summary = create_summary_table(results, correlations)
    
    # Save summary
    output_path = "results_imagenet_stats/pca_norm_comparison_metrics.json"
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Saved comparison summary to {output_path}")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY COMPARISON")
    logger.info("="*80)
    for norm in norms:
        if norm in summary:
            logger.info(f"\n{norm.upper()}:")
            logger.info(f"  Components for 95% variance: {summary[norm]['n_components_95pct']}")
            logger.info(f"  First PC variance: {summary[norm]['first_pc_variance']:.4f}")
            logger.info(f"  Top 5 cumulative variance: {summary[norm]['top5_cumulative_variance']:.4f}")
            if summary[norm]['silhouette_10comp'] is not None:
                logger.info(f"  Silhouette (10 comp): {summary[norm]['silhouette_10comp']:.4f}")
                logger.info(f"  Davies-Bouldin (10 comp): {summary[norm]['davies_bouldin_10comp']:.4f}")
                logger.info(f"  Calinski-Harabasz (10 comp): {summary[norm]['calinski_harabasz_10comp']:.2f}")
            logger.info(f"  Separability ratio: {summary[norm]['separability_ratio']:.4f}")
    
    logger.info(f"\nCORRELATIONS:")
    for pair, corr in summary['correlations'].items():
        logger.info(f"  {pair}: {corr:.4f}")
    
    logger.info("\n" + "="*80)
    
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare PCA results across norms')
    parser.add_argument('--norms', nargs='+', default=['l1', 'l2', 'linf'],
                      help='List of norms to compare')
    parser.add_argument('--eval-dir', type=str, default="/mnt/data/datasets/imagenet/val/",
                      help='Path to ImageNet validation directory')
    
    args = parser.parse_args()
    
    # Load labels
    from torchvision import datasets, transforms
    logger.info("Loading ImageNet labels...")
    dataset_eval = datasets.ImageFolder(
        root=args.eval_dir,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    )
    labels = np.array([s[1] for s in dataset_eval.samples], dtype=np.int32)
    logger.info(f"Loaded {len(labels)} labels")
    
    # Run comparison
    summary = compare_norms(args.norms, labels)
    
    logger.info("\nComparison complete!")
