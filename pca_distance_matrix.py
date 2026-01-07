import torch
import numpy as np
from sklearn.decomposition import IncrementalPCA
import pickle
import json
import logging
import os
import argparse

# Set up logging
os.makedirs("logs", exist_ok=True)
log_file = os.path.join("logs", "pca_distance_matrix.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

def apply_pca_to_distance_matrix(
    dist_matrix_path,
    output_transformed_path,
    output_model_path,
    output_metadata_path,
    variance_threshold=0.95,
    batch_size=1000
):
    """
    Apply PCA to a distance matrix using incremental PCA to handle memory constraints.
    
    Args:
        dist_matrix_path: Path to distance matrix .pt file
        output_transformed_path: Path to save transformed data
        output_model_path: Path to save PCA model
        output_metadata_path: Path to save metadata JSON
        variance_threshold: Cumulative variance to capture (default 0.95)
        batch_size: Batch size for incremental PCA
    """
    logger.info(f"Loading distance matrix from {dist_matrix_path}")
    dist_matrix = torch.load(dist_matrix_path)
    
    if isinstance(dist_matrix, torch.Tensor):
        dist_matrix = dist_matrix.cpu().numpy()
    
    n_samples = dist_matrix.shape[0]
    logger.info(f"Distance matrix shape: {dist_matrix.shape}")
    logger.info(f"Distance matrix memory: {dist_matrix.nbytes / (1024**3):.2f} GB")
    
    # Use distance matrix rows as features (each row is a 50000-dim feature vector)
    X = dist_matrix
    
    # First pass: determine number of components needed for variance threshold
    logger.info("First pass: determining number of components...")
    pca_full = IncrementalPCA(n_components=min(n_samples, 1000), batch_size=batch_size)
    
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        pca_full.partial_fit(X[i:batch_end])
        if (i // batch_size) % 10 == 0:
            logger.info(f"  Fitted batch {i//batch_size + 1}/{(n_samples + batch_size - 1)//batch_size}")
    
    cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.searchsorted(cumsum_var, variance_threshold) + 1
    n_components = min(n_components, len(pca_full.explained_variance_ratio_))
    
    logger.info(f"Number of components for {variance_threshold*100}% variance: {n_components}")
    logger.info(f"Actual variance captured: {cumsum_var[n_components-1]*100:.2f}%")
    
    # Second pass: fit with optimal number of components and transform
    logger.info(f"Second pass: fitting PCA with {n_components} components...")
    pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        pca.partial_fit(X[i:batch_end])
        if (i // batch_size) % 10 == 0:
            logger.info(f"  Fitted batch {i//batch_size + 1}/{(n_samples + batch_size - 1)//batch_size}")
    
    # Transform the data
    logger.info("Transforming data...")
    X_transformed = np.zeros((n_samples, n_components), dtype=np.float32)
    
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        X_transformed[i:batch_end] = pca.transform(X[i:batch_end])
        if (i // batch_size) % 10 == 0:
            logger.info(f"  Transformed batch {i//batch_size + 1}/{(n_samples + batch_size - 1)//batch_size}")
    
    # Save transformed data
    logger.info(f"Saving transformed data to {output_transformed_path}")
    torch.save(torch.from_numpy(X_transformed), output_transformed_path)
    
    # Save PCA model
    logger.info(f"Saving PCA model to {output_model_path}")
    with open(output_model_path, 'wb') as f:
        pickle.dump(pca, f)
    
    # Save metadata
    metadata = {
        'n_components': int(n_components),
        'explained_variance': pca.explained_variance_.tolist(),
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
        'singular_values': pca.singular_values_.tolist(),
        'n_samples': int(n_samples),
        'n_features': int(X.shape[1]),
        'variance_threshold': variance_threshold
    }
    
    logger.info(f"Saving metadata to {output_metadata_path}")
    with open(output_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"PCA complete! Reduced from {X.shape[1]} to {n_components} dimensions")
    logger.info(f"Top 10 explained variance ratios: {pca.explained_variance_ratio_[:10]}")
    
    return metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply PCA to distance matrices')
    parser.add_argument('--norms', nargs='+', default=['l1', 'l2', 'linf'],
                      help='List of norms to process')
    parser.add_argument('--variance', type=float, default=0.95,
                      help='Variance threshold (default: 0.95)')
    parser.add_argument('--batch-size', type=int, default=1000,
                      help='Batch size for incremental PCA (default: 1000)')
    
    args = parser.parse_args()
    
    for norm in args.norms:
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing norm: {norm}")
        logger.info(f"{'='*80}\n")
        
        dist_matrix_path = f"results_imagenet_stats/val_dist_matrix_{norm}.pt"
        output_transformed_path = f"results_imagenet_stats/val_pca_transformed_{norm}.pt"
        output_model_path = f"results_imagenet_stats/val_pca_model_{norm}.pkl"
        output_metadata_path = f"results_imagenet_stats/val_pca_metadata_{norm}.json"
        
        try:
            apply_pca_to_distance_matrix(
                dist_matrix_path=dist_matrix_path,
                output_transformed_path=output_transformed_path,
                output_model_path=output_model_path,
                output_metadata_path=output_metadata_path,
                variance_threshold=args.variance,
                batch_size=args.batch_size
            )
        except Exception as e:
            logger.error(f"Error processing {norm}: {e}", exc_info=True)
            continue
    
    logger.info("\nAll norms processed successfully!")
