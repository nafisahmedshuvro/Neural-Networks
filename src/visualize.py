"""
Visualization Script
Creates t-SNE and UMAP visualizations of latent space and clusters
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap

def visualize_latent_space(latent_features, cluster_labels, method='tsne', output_path=None):
    """
    Visualize latent space using t-SNE or UMAP
    
    Args:
        latent_features: Latent feature array
        cluster_labels: Cluster assignments
        method: Visualization method ('tsne' or 'umap')
        output_path: Path to save figure
    """
    print(f"Computing {method.upper()} embedding...")
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        embedding = reducer.fit_transform(latent_features)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(latent_features)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                         c=cluster_labels, cmap='tab10', 
                         alpha=0.6, s=50)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'Latent Space Visualization ({method.upper()})')
    plt.xlabel(f'{method.upper()} Dimension 1')
    plt.ylabel(f'{method.upper()} Dimension 2')
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize latent space")
    parser.add_argument("--latent_path", type=str, required=True, help="Path to latent features")
    parser.add_argument("--clusters_path", type=str, required=True, help="Path to cluster labels")
    parser.add_argument("--method", type=str, choices=['tsne', 'umap'], default='tsne',
                       help="Visualization method")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for figure")
    
    args = parser.parse_args()
    
    # Load data
    latent_features = np.load(args.latent_path)
    cluster_labels = np.load(args.clusters_path)
    
    print(f"Visualizing {len(latent_features)} samples with {len(np.unique(cluster_labels))} clusters")
    
    # Visualize
    visualize_latent_space(latent_features, cluster_labels, 
                          method=args.method, output_path=args.output_path)



