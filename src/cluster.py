"""
Clustering Script
Performs K-Means, Agglomerative, or DBSCAN clustering on latent features
"""
import os
import argparse
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import pickle

def perform_clustering(latent_features, method='kmeans', n_clusters=8, **kwargs):
    """
    Perform clustering on latent features
    
    Args:
        latent_features: Latent feature array (n_samples, n_features)
        method: Clustering method ('kmeans', 'agglomerative', 'dbscan')
        n_clusters: Number of clusters (for kmeans, agglomerative)
        **kwargs: Additional parameters for clustering algorithms
    """
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(latent_features)
        
    elif method == 'agglomerative':
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = clusterer.fit_predict(latent_features)
        
    elif method == 'dbscan':
        eps = kwargs.get('eps', 0.5)
        min_samples = kwargs.get('min_samples', 5)
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clusterer.fit_predict(latent_features)
        
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    return labels, clusterer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform clustering on latent features")
    parser.add_argument("--latent_path", type=str, required=True, help="Path to latent features .npy file")
    parser.add_argument("--method", type=str, choices=['kmeans', 'agglomerative', 'dbscan'],
                       default='kmeans', help="Clustering method")
    parser.add_argument("--n_clusters", type=int, default=8, help="Number of clusters")
    parser.add_argument("--eps", type=float, default=0.5, help="DBSCAN eps parameter")
    parser.add_argument("--min_samples", type=int, default=5, help="DBSCAN min_samples parameter")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for cluster labels")
    
    args = parser.parse_args()
    
    # Load latent features
    latent_features = np.load(args.latent_path)
    print(f"Loaded latent features: {latent_features.shape}")
    
    # Perform clustering
    kwargs = {}
    if args.method == 'dbscan':
        kwargs['eps'] = args.eps
        kwargs['min_samples'] = args.min_samples
    
    labels, clusterer = perform_clustering(
        latent_features,
        method=args.method,
        n_clusters=args.n_clusters,
        **kwargs
    )
    
    # Save cluster labels
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.save(args.output_path, labels)
    
    # Print cluster statistics
    unique_labels = np.unique(labels)
    print(f"\nClustering complete using {args.method}")
    print(f"Number of clusters found: {len(unique_labels)}")
    print(f"Cluster sizes: {np.bincount(labels + 1) if args.method == 'dbscan' else np.bincount(labels)}")
    print(f"Labels saved to: {args.output_path}")



