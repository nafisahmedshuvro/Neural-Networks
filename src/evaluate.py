"""
Clustering Evaluation Script
Computes clustering metrics: Silhouette, CH Index, DB Index, ARI, NMI, Purity
"""
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score
)

def cluster_purity(y_true, y_pred):
    """
    Compute cluster purity
    
    Args:
        y_true: True labels
        y_pred: Predicted cluster labels
    """
    n = len(y_true)
    clusters = np.unique(y_pred)
    purity = 0
    
    for cluster in clusters:
        cluster_mask = y_pred == cluster
        cluster_labels = y_true[cluster_mask]
        if len(cluster_labels) > 0:
            most_common = np.bincount(cluster_labels).max()
            purity += most_common
    
    return purity / n

def evaluate_clustering(latent_features, cluster_labels, true_labels=None):
    """
    Evaluate clustering quality using multiple metrics
    
    Args:
        latent_features: Latent feature array
        cluster_labels: Cluster assignments
        true_labels: True labels (optional, for supervised metrics)
    """
    metrics = {}
    
    # Unsupervised metrics
    try:
        metrics['silhouette_score'] = silhouette_score(latent_features, cluster_labels)
    except:
        metrics['silhouette_score'] = np.nan
    
    try:
        metrics['calinski_harabasz'] = calinski_harabasz_score(latent_features, cluster_labels)
    except:
        metrics['calinski_harabasz'] = np.nan
    
    try:
        metrics['davies_bouldin'] = davies_bouldin_score(latent_features, cluster_labels)
    except:
        metrics['davies_bouldin'] = np.nan
    
    # Supervised metrics (if labels available)
    if true_labels is not None:
        try:
            metrics['adjusted_rand_index'] = adjusted_rand_score(true_labels, cluster_labels)
        except:
            metrics['adjusted_rand_index'] = np.nan
        
        try:
            metrics['normalized_mutual_info'] = normalized_mutual_info_score(true_labels, cluster_labels)
        except:
            metrics['normalized_mutual_info'] = np.nan
        
        try:
            metrics['purity'] = cluster_purity(true_labels, cluster_labels)
        except:
            metrics['purity'] = np.nan
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate clustering results")
    parser.add_argument("--latent_path", type=str, required=True, help="Path to latent features")
    parser.add_argument("--clusters_path", type=str, required=True, help="Path to cluster labels")
    parser.add_argument("--labels_path", type=str, default=None, help="Path to true labels (optional)")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for metrics CSV")
    
    args = parser.parse_args()
    
    # Load data
    latent_features = np.load(args.latent_path)
    cluster_labels = np.load(args.clusters_path)
    
    true_labels = None
    if args.labels_path and os.path.exists(args.labels_path):
        if args.labels_path.endswith('.csv'):
            df = pd.read_csv(args.labels_path)
            true_labels = df['label'].values if 'label' in df.columns else df.iloc[:, 0].values
        else:
            true_labels = np.load(args.labels_path)
    
    print(f"Evaluating clustering on {len(latent_features)} samples")
    print(f"Number of clusters: {len(np.unique(cluster_labels))}")
    
    # Evaluate
    metrics = evaluate_clustering(latent_features, cluster_labels, true_labels)
    
    # Print results
    print("\n=== Clustering Metrics ===")
    for metric_name, value in metrics.items():
        if not np.isnan(value):
            print(f"{metric_name}: {value:.4f}")
    
    # Save to CSV
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df_metrics = pd.DataFrame([metrics])
    df_metrics.to_csv(args.output_path, index=False)
    print(f"\nMetrics saved to: {args.output_path}")



