"""
Execute Easy Task: Basic VAE + K-Means Clustering
This script runs the complete Easy Task workflow
"""
import os
import sys
import subprocess
import argparse

def run_command(cmd, description, allow_skip=False):
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
        print(result.stderr[:500])  # Print first 500 chars of error
        
        # Check for PyTorch DLL error
        if "OSError" in result.stderr and "DLL" in result.stderr:
            print("\n[WARN] PyTorch DLL Error Detected!")
            print("This is a Windows compatibility issue.")
            print("\nSolutions:")
            print("1. Install Visual C++ Redistributables:")
            print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
            print("2. Or use alternative approach (see FIX_PYTORCH_ERROR.md)")
            
            if allow_skip:
                print("\n[WARN] Skipping this step (allow_skip=True)")
                return "skip"
        
        return False
    
    print(f"\n[OK] Completed: {description}")
    if result.stdout:
        print(result.stdout[-500:])  # Print last 500 chars of output
    return True

def check_prerequisites():
    """Check if required data and files exist"""
    print("Checking prerequisites...")
    
    issues = []
    
    # Check for audio data
    audio_files = []
    if os.path.exists("project/data/gtzan/genres_original"):
        import glob
        audio_files = glob.glob("project/data/gtzan/genres_original/**/*.wav", recursive=True)
    
    if not audio_files:
        # Check alternative locations
        import glob
        for ext in ['*.wav', '*.mp3', '*.flac']:
            audio_files.extend(glob.glob(f"project/data/**/{ext}", recursive=True))
    
    if not audio_files:
        issues.append("No audio files found. Need GTZAN dataset or audio files.")
    else:
        print(f"[OK] Found {len(audio_files)} audio files")
    
    # Check for lyrics
    if os.path.exists("project/data/song_lyrics.csv"):
        print("[OK] song_lyrics.csv found")
    else:
        print("[WARN] song_lyrics.csv not found (optional for Easy Task)")
    
    # Check PyTorch availability (non-blocking - we have fallback)
    print("Checking PyTorch...")
    pytorch_available = False
    try:
        import torch
        print(f"[OK] PyTorch {torch.__version__} available")
        pytorch_available = True
    except Exception as e:
        error_msg = str(e)
        if "DLL" in error_msg or "c10.dll" in error_msg:
            print("[WARN] PyTorch DLL error detected - will use fallback method")
            print("   (scikit-learn autoencoder will be used instead)")
        else:
            print(f"[WARN] PyTorch error: {error_msg[:100]}")
            print("   (scikit-learn autoencoder will be used instead)")
    
    # Store PyTorch availability for later use
    return len(issues) == 0, issues, pytorch_available

def execute_easy_task():
    """Execute the complete Easy Task workflow"""
    
    print("="*60)
    print("EASY TASK EXECUTION: Basic VAE + K-Means Clustering")
    print("="*60)
    print()
    
    # Check prerequisites
    ready, issues, pytorch_available = check_prerequisites()
    
    if not ready:
        print("\n[ERROR] Prerequisites not met:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease ensure:")
        print("  1. GTZAN dataset is downloaded to: project/data/gtzan/")
        print("  2. Or audio files are available in project/data/")
        return False
    
    if not pytorch_available:
        print("\n[INFO] PyTorch not available - will use scikit-learn fallback")
        print("       This will use a simpler autoencoder instead of full VAE")
    
    # Step 1: Preprocess datasets
    if not run_command(
        'python project/src/preprocess_datasets.py --gtzan_dir project/data/gtzan --output_dir project/data/processed --duration 30',
        "[Step 1/8] Preprocessing datasets"
    ):
        return False
    
    # Step 2: Extract MFCC features
    if not run_command(
        'python project/src/extract_features.py --audio_dir project/data/processed/gtzan/audio --output_dir project/data/features/audio --feature_type mfcc --n_mfcc 13',
        "[Step 2/8] Extracting MFCC features"
    ):
        return False
    
    # Step 3: Train Basic VAE or Fallback
    step3_result = None
    if pytorch_available:
        print("\nAttempting to train VAE with PyTorch...")
        step3_result = run_command(
            'python project/src/train_vae.py --data_path project/data/features/audio/mfcc_features.npy --model_type basic --latent_dim 32 --epochs 100 --batch_size 32 --save_path project/models/vae_basic.pth',
            "[Step 3/8] Training Basic VAE",
            allow_skip=False
        )
    
    if not pytorch_available or step3_result == False:
        if not pytorch_available:
            print("\n" + "="*60)
            print("Using fallback method (scikit-learn autoencoder)...")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("PyTorch failed. Trying fallback method (scikit-learn)...")
            print("="*60)
        
        # Try fallback method (use latent_dim=10 since we only have 13 features)
        fallback_result = run_command(
            'python project/src/train_vae_fallback.py --data_path project/data/features/audio/mfcc_features.npy --latent_dim 10 --epochs 100 --save_path project/models/vae_sklearn.pkl',
            "[Step 3/8] Training Autoencoder (Fallback - scikit-learn)",
            allow_skip=False
        )
        
        if fallback_result == False:
            print("\n[ERROR] Fallback method failed!")
            print("Cannot proceed without a trained model.")
            return False
        else:
            # Update paths for fallback model
            print("\n[INFO] Using scikit-learn model. Extracting latent features...")
            import numpy as np
            import pickle
            os.makedirs('project/data/latent', exist_ok=True)
            with open('project/models/vae_sklearn.pkl', 'rb') as f:
                model_data = pickle.load(f)
            np.save('project/data/latent/latent_features.npy', model_data['latent_features'])
            print("[OK] Latent features saved: project/data/latent/latent_features.npy")
            # Skip Step 4 since we already have latent features
            step3_result = "skip_step4"
    
    # Step 4: Extract latent features
    if step3_result == "skip_step4":
        print("\n[SKIP] Step 4 - Latent features already extracted from sklearn model")
    else:
        if not run_command(
            'python project/src/extract_latent.py --model_path project/models/vae_basic.pth --data_path project/data/features/audio/mfcc_features.npy --model_type basic --latent_dim 32 --output_path project/data/latent/latent_features.npy',
            "[Step 4/8] Extracting latent features"
        ):
            return False
    
    # Step 5: K-Means clustering (use 10 clusters for 10 genres)
    if not run_command(
        'python project/src/cluster.py --latent_path project/data/latent/latent_features.npy --method kmeans --n_clusters 10 --output_path project/results/clusters_kmeans.npy',
        "[Step 5/8] Performing K-Means clustering"
    ):
        return False
    
    # Step 6: Evaluate clustering
    if not run_command(
        'python project/src/evaluate.py --latent_path project/data/latent/latent_features.npy --clusters_path project/results/clusters_kmeans.npy --output_path project/results/metrics_kmeans.csv',
        "[Step 6/8] Evaluating clustering metrics"
    ):
        return False
    
    # Step 7: Visualize with t-SNE
    if not run_command(
        'python project/src/visualize.py --latent_path project/data/latent/latent_features.npy --clusters_path project/results/clusters_kmeans.npy --method tsne --output_path project/results/latent_visualization/tsne_plot.png',
        "[Step 7/8] Creating t-SNE visualization"
    ):
        return False
    
    # Step 8: Baseline comparison (PCA + K-Means)
    # Use n_components=10 since we only have 13 features
    if not run_command(
        'python project/src/baseline_clustering.py --data_path project/data/features/audio/mfcc_features.npy --method pca_kmeans --n_components 10 --n_clusters 10 --output_path project/results/clusters_pca_kmeans.npy',
        "[Step 8/8] Baseline: PCA + K-Means"
    ):
        return False
    
    # Evaluate baseline
    run_command(
        'python project/src/evaluate.py --latent_path project/data/features/audio/mfcc_features.npy --clusters_path project/results/clusters_pca_kmeans.npy --output_path project/results/metrics_pca_kmeans.csv',
        "Evaluating baseline metrics"
    )
    
    print("\n" + "="*60)
    print("[SUCCESS] EASY TASK COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nResults saved in:")
    print("  - Model: project/models/vae_basic.pth")
    print("  - Latent features: project/data/latent/latent_features.npy")
    print("  - Clusters: project/results/clusters_kmeans.npy")
    print("  - Metrics: project/results/metrics_kmeans.csv")
    print("  - Visualization: project/results/latent_visualization/tsne_plot.png")
    print("  - Baseline metrics: project/results/metrics_pca_kmeans.csv")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute Easy Task workflow")
    args = parser.parse_args()
    
    success = execute_easy_task()
    sys.exit(0 if success else 1)

