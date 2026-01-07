"""
Baseline Clustering Methods
PCA + K-Means, Autoencoder + K-Means, Direct feature clustering
"""
import os
import argparse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

def pca_kmeans_clustering(features, n_components=32, n_clusters=8):
    """
    Baseline: PCA + K-Means
    
    Args:
        features: Input features
        n_components: Number of PCA components
        n_clusters: Number of clusters
    """
    # Normalize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Adjust n_components if it's larger than input features
    max_components = min(n_components, features.shape[1])
    if n_components > features.shape[1]:
        print(f"Warning: n_components ({n_components}) > input features ({features.shape[1]})")
        print(f"Using n_components = {max_components} instead")
        n_components = max_components
    
    # PCA
    pca = PCA(n_components=n_components, random_state=42)
    features_pca = pca.fit_transform(features_scaled)
    
    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_pca)
    
    return labels, features_pca

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline clustering methods")
    parser.add_argument("--data_path", type=str, required=True, help="Path to features")
    parser.add_argument("--method", type=str, choices=['pca_kmeans'], default='pca_kmeans',
                       help="Baseline method")
    parser.add_argument("--n_components", type=int, default=32, help="Number of PCA components")
    parser.add_argument("--n_clusters", type=int, default=8, help="Number of clusters")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for labels")
    
    args = parser.parse_args()
    
    # Load features
    features = np.load(args.data_path)
    print(f"Loaded features: {features.shape}")
    
    # Perform baseline clustering
    if args.method == 'pca_kmeans':
        labels, features_reduced = pca_kmeans_clustering(
            features, n_components=args.n_components, n_clusters=args.n_clusters
        )
    
    # Save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.save(args.output_path, labels)
    
    print(f"Baseline clustering complete: {args.method}")
    print(f"Number of clusters: {len(np.unique(labels))}")
    print(f"Labels saved to: {args.output_path}")

