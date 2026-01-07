"""
Execute Medium Task: Enhanced VAE + Hybrid Features + Multiple Clustering
This script runs the complete Medium Task workflow
"""
import os
import sys
import subprocess
import argparse

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}")
    print()
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"\n[ERROR] Error in: {description}")
        print("Error output:")
        print(result.stderr[:500])
        return False
    
    print(f"\n[OK] Completed: {description}")
    if result.stdout:
        print(result.stdout[-300:])
    return True

def check_prerequisites():
    """Check if required data and files exist"""
    print("Checking prerequisites for Medium Task...")
    
    issues = []
    
    # Check for audio features
    if os.path.exists("project/data/features/audio/mfcc_features.npy"):
        print("[OK] MFCC features found")
    else:
        issues.append("MFCC features not found. Run Easy Task first or extract features.")
    
    # Check for lyrics data
    if os.path.exists("project/data/song_lyrics.csv"):
        print("[OK] song_lyrics.csv found")
    else:
        issues.append("song_lyrics.csv not found")
    
    # Check for processed audio
    if os.path.exists("project/data/processed/gtzan/audio"):
        audio_files = len([f for f in os.listdir("project/data/processed/gtzan/audio") if f.endswith('.wav')])
        if audio_files > 0:
            print(f"[OK] Processed audio files found: {audio_files}")
        else:
            issues.append("No processed audio files found")
    else:
        issues.append("Processed audio directory not found")
    
    return len(issues) == 0, issues

def execute_medium_task():
    """Execute the complete Medium Task workflow"""
    
    print("="*60)
    print("MEDIUM TASK EXECUTION: Enhanced VAE + Hybrid Features")
    print("="*60)
    print()
    
    # Check prerequisites
    ready, issues = check_prerequisites()
    
    if not ready:
        print("\n[ERROR] Prerequisites not met:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    # Step 1: Extract Mel-spectrograms (for Convolutional VAE)
    if not run_command(
        'python project/src/extract_features.py --audio_dir project/data/processed/gtzan/audio --output_dir project/data/features/spectrograms --feature_type melspectrogram --n_mels 128',
        "[Step 1/10] Extracting Mel-spectrograms for Convolutional VAE"
    ):
        return False
    
    # Step 2: Extract lyrics embeddings
    if not run_command(
        'python project/src/extract_lyrics_features.py --lyrics_csv project/data/song_lyrics.csv --output_path project/data/features/lyrics/lyrics_embeddings.npy --model_name all-MiniLM-L6-v2',
        "[Step 2/10] Extracting lyrics embeddings"
    ):
        print("[WARN] Lyrics extraction failed - continuing with audio-only")
        use_hybrid = False
    else:
        use_hybrid = True
    
    # Step 3: Combine audio + lyrics features (if lyrics available)
    if use_hybrid:
        if not run_command(
            'python project/src/extract_hybrid_features.py --audio_features project/data/features/audio/mfcc_features.npy --lyrics_features project/data/features/lyrics/lyrics_embeddings.npy --output_path project/data/features/hybrid/hybrid_features.npy --fusion_method concatenate',
            "[Step 3/10] Combining audio + lyrics features"
        ):
            print("[WARN] Hybrid feature combination failed - using audio-only")
            use_hybrid = False
    
    # Step 4: Train Convolutional VAE on spectrograms
    print("\n[INFO] Training Convolutional VAE on Mel-spectrograms...")
    if not run_command(
        'python project/src/train_vae.py --data_path project/data/features/spectrograms/melspectrogram_features.npy --model_type conv --latent_dim 32 --epochs 150 --batch_size 16 --save_path project/models/vae_conv.pth',
        "[Step 4/10] Training Convolutional VAE"
    ):
        print("[WARN] PyTorch failed - using sklearn fallback for spectrograms")
        # Use PCA for spectrograms as fallback
        if not run_command(
            'python project/src/train_vae_fallback.py --data_path project/data/features/spectrograms/melspectrogram_features.npy --latent_dim 32 --epochs 100 --save_path project/models/vae_conv_sklearn.pkl',
            "[Step 4/10] Training Autoencoder (Fallback for spectrograms)"
        ):
            return False
        conv_model_type = "sklearn"
    else:
        conv_model_type = "pytorch"
    
    # Step 5: Extract latent features from Convolutional VAE
    if conv_model_type == "pytorch":
        if not run_command(
            'python project/src/extract_latent.py --model_path project/models/vae_conv.pth --data_path project/data/features/spectrograms/melspectrogram_features.npy --model_type conv --latent_dim 32 --output_path project/data/latent/latent_conv.npy',
            "[Step 5/10] Extracting latent features from Conv VAE"
        ):
            return False
    else:
        # Extract from sklearn model
        import numpy as np
        import pickle
        with open('project/models/vae_conv_sklearn.pkl', 'rb') as f:
            model_data = pickle.load(f)
        os.makedirs('project/data/latent', exist_ok=True)
        np.save('project/data/latent/latent_conv.npy', model_data['latent_features'])
        print("[OK] Latent features extracted from sklearn model")
    
    # Step 6: Train VAE on hybrid features (if available)
    if use_hybrid:
        if not run_command(
            'python project/src/train_vae.py --data_path project/data/features/hybrid/hybrid_features.npy --model_type basic --latent_dim 64 --epochs 150 --batch_size 32 --save_path project/models/vae_hybrid.pth',
            "[Step 6/10] Training VAE on hybrid features"
        ):
            print("[WARN] PyTorch failed - using sklearn fallback for hybrid")
            if not run_command(
                'python project/src/train_vae_fallback.py --data_path project/data/features/hybrid/hybrid_features.npy --latent_dim 32 --epochs 100 --save_path project/models/vae_hybrid_sklearn.pkl',
                "[Step 6/10] Training Autoencoder on hybrid features (Fallback)"
            ):
                use_hybrid = False
            else:
                # Extract latent from sklearn
                import numpy as np
                import pickle
                with open('project/models/vae_hybrid_sklearn.pkl', 'rb') as f:
                    model_data = pickle.load(f)
                os.makedirs('project/data/latent', exist_ok=True)
                np.save('project/data/latent/latent_hybrid.npy', model_data['latent_features'])
                print("[OK] Hybrid latent features extracted")
    
    # Step 7: Multiple Clustering Algorithms
    latent_paths = [
        ("project/data/latent/latent_conv.npy", "conv"),
    ]
    
    if use_hybrid:
        latent_paths.append(("project/data/latent/latent_hybrid.npy", "hybrid"))
    
    for latent_path, name in latent_paths:
        # K-Means
        run_command(
            f'python project/src/cluster.py --latent_path {latent_path} --method kmeans --n_clusters 10 --output_path project/results/clusters_{name}_kmeans.npy',
            f"[Step 7/10] K-Means clustering ({name})"
        )
        
        # Agglomerative Clustering
        run_command(
            f'python project/src/cluster.py --latent_path {latent_path} --method agglomerative --n_clusters 10 --output_path project/results/clusters_{name}_agglomerative.npy',
            f"[Step 7/10] Agglomerative clustering ({name})"
        )
        
        # DBSCAN
        run_command(
            f'python project/src/cluster.py --latent_path {latent_path} --method dbscan --eps 0.5 --min_samples 5 --output_path project/results/clusters_{name}_dbscan.npy',
            f"[Step 7/10] DBSCAN clustering ({name})"
        )
    
    # Step 8: Evaluate all clustering results
    print("\n[INFO] Evaluating all clustering methods...")
    for latent_path, name in latent_paths:
        for method in ['kmeans', 'agglomerative', 'dbscan']:
            cluster_path = f"project/results/clusters_{name}_{method}.npy"
            if os.path.exists(cluster_path):
                run_command(
                    f'python project/src/evaluate.py --latent_path {latent_path} --clusters_path {cluster_path} --output_path project/results/metrics_{name}_{method}.csv',
                    f"[Step 8/10] Evaluating {name} + {method}"
                )
    
    # Step 9: Visualizations
    for latent_path, name in latent_paths:
        for method in ['kmeans', 'agglomerative']:
            cluster_path = f"project/results/clusters_{name}_{method}.npy"
            if os.path.exists(cluster_path):
                # t-SNE
                run_command(
                    f'python project/src/visualize.py --latent_path {latent_path} --clusters_path {cluster_path} --method tsne --output_path project/results/latent_visualization/{name}_{method}_tsne.png',
                    f"[Step 9/10] t-SNE visualization ({name} + {method})"
                )
                # UMAP
                run_command(
                    f'python project/src/visualize.py --latent_path {latent_path} --clusters_path {cluster_path} --method umap --output_path project/results/latent_visualization/{name}_{method}_umap.png',
                    f"[Step 9/10] UMAP visualization ({name} + {method})"
                )
    
    # Step 10: Compare with baselines
    print("\n[INFO] Comparing with baseline methods...")
    run_command(
        'python project/src/baseline_clustering.py --data_path project/data/features/audio/mfcc_features.npy --method pca_kmeans --n_components 10 --n_clusters 10 --output_path project/results/clusters_baseline_pca_kmeans.npy',
        "[Step 10/10] Baseline: PCA + K-Means"
    )
    
    if os.path.exists("project/data/features/spectrograms/melspectrogram_features.npy"):
        # Flatten spectrograms for baseline
        import numpy as np
        spec_features = np.load("project/data/features/spectrograms/melspectrogram_features.npy")
        spec_flat = spec_features.reshape(spec_features.shape[0], -1)
        flat_path = "project/data/features/spectrograms/melspectrogram_flat.npy"
        np.save(flat_path, spec_flat)
        
        run_command(
            f'python project/src/baseline_clustering.py --data_path {flat_path} --method pca_kmeans --n_components 32 --n_clusters 10 --output_path project/results/clusters_baseline_spec_pca_kmeans.npy',
            "[Step 10/10] Baseline: Spectrogram PCA + K-Means"
        )
    
    print("\n" + "="*60)
    print("[SUCCESS] MEDIUM TASK COMPLETED!")
    print("="*60)
    print("\nResults saved in:")
    print("  - Models: project/models/")
    print("  - Latent features: project/data/latent/")
    print("  - Clusters: project/results/clusters_*.npy")
    print("  - Metrics: project/results/metrics_*.csv")
    print("  - Visualizations: project/results/latent_visualization/")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute Medium Task workflow")
    args = parser.parse_args()
    
    success = execute_medium_task()
    sys.exit(0 if success else 1)



