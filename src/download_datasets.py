"""
Dataset Download Script
Downloads and prepares GTZAN, Genius Lyrics, and DALI datasets
"""
import os
import argparse
import subprocess
import json
import requests
import zipfile
from pathlib import Path

def download_kaggle_dataset(dataset_name, output_dir, unzip=True):
    """
    Download dataset from Kaggle using Kaggle API
    
    Args:
        dataset_name: Kaggle dataset identifier (e.g., 'andradaolteanu/gtzan-dataset-music-genre-classification')
        output_dir: Directory to save dataset
        unzip: Whether to unzip the downloaded file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading {dataset_name} from Kaggle...")
    print("Note: Requires Kaggle API credentials (kaggle.json)")
    
    try:
        # Use Kaggle CLI
        cmd = f'kaggle datasets download -d {dataset_name} -p "{output_dir}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error downloading {dataset_name}: {result.stderr}")
            print("\nTo use Kaggle API:")
            print("1. Install: pip install kaggle")
            print("2. Get API token from: https://www.kaggle.com/account")
            print("3. Place kaggle.json in: C:\\Users\\<YourUsername>\\.kaggle\\")
            return False
        
        # Unzip if needed
        if unzip:
            zip_files = list(Path(output_dir).glob("*.zip"))
            for zip_file in zip_files:
                print(f"Extracting {zip_file.name}...")
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
                # Optionally remove zip file
                # zip_file.unlink()
        
        print(f"Successfully downloaded {dataset_name} to {output_dir}")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def download_dali_dataset(output_dir):
    """
    Download DALI dataset from GitHub
    
    Args:
        output_dir: Directory to save DALI dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading DALI dataset...")
    print("DALI dataset can be downloaded from:")
    print("https://github.com/gabolsgabs/DALI")
    print("\nFor automated download, you can use git:")
    
    dali_repo = "https://github.com/gabolsgabs/DALI.git"
    dali_path = os.path.join(output_dir, "DALI")
    
    if os.path.exists(dali_path):
        print(f"DALI already exists at {dali_path}")
        return True
    
    try:
        # Clone repository
        cmd = f'git clone {dali_repo} "{dali_path}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error cloning DALI: {result.stderr}")
            print("\nManual download:")
            print("1. Visit: https://github.com/gabolsgabs/DALI")
            print("2. Download or clone the repository")
            print(f"3. Place it in: {dali_path}")
            return False
        
        print(f"Successfully cloned DALI to {dali_path}")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def setup_datasets():
    """
    Setup all three datasets
    """
    base_dir = "project/data"
    
    datasets = {
        "gtzan": {
            "kaggle_id": "andradaolteanu/gtzan-dataset-music-genre-classification",
            "output_dir": os.path.join(base_dir, "gtzan"),
            "description": "GTZAN Genre Classification Dataset"
        },
        "genius_lyrics": {
            "kaggle_id": "carlosgdcj/genius-song-lyrics-with-language-information",
            "output_dir": os.path.join(base_dir, "genius_lyrics"),
            "description": "Genius Song Lyrics with Language Information"
        }
    }
    
    print("=" * 60)
    print("Dataset Download Setup")
    print("=" * 60)
    print("\nThis script will help you download:")
    print("1. GTZAN Dataset (Audio + Genre labels)")
    print("2. Genius Lyrics Dataset (Lyrics + Language info)")
    print("3. DALI Dataset (Audio + Lyrics + Vocal notes)")
    print("\n" + "=" * 60)
    
    # Download GTZAN
    print("\n[1/3] GTZAN Dataset")
    download_kaggle_dataset(
        datasets["gtzan"]["kaggle_id"],
        datasets["gtzan"]["output_dir"]
    )
    
    # Download Genius Lyrics
    print("\n[2/3] Genius Lyrics Dataset")
    download_kaggle_dataset(
        datasets["genius_lyrics"]["kaggle_id"],
        datasets["genius_lyrics"]["output_dir"]
    )
    
    # Download DALI
    print("\n[3/3] DALI Dataset")
    download_dali_dataset(os.path.join(base_dir, "dali"))
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run: python project/src/preprocess_datasets.py")
    print("2. This will process and combine all datasets")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument("--dataset", type=str, choices=['gtzan', 'genius', 'dali', 'all'],
                       default='all', help="Which dataset to download")
    parser.add_argument("--output_base", type=str, default="project/data",
                       help="Base directory for datasets")
    
    args = parser.parse_args()
    
    if args.dataset == 'all':
        setup_datasets()
    elif args.dataset == 'gtzan':
        download_kaggle_dataset(
            "andradaolteanu/gtzan-dataset-music-genre-classification",
            os.path.join(args.output_base, "gtzan")
        )
    elif args.dataset == 'genius':
        download_kaggle_dataset(
            "carlosgdcj/genius-song-lyrics-with-language-information",
            os.path.join(args.output_base, "genius_lyrics")
        )
    elif args.dataset == 'dali':
        download_dali_dataset(os.path.join(args.output_base, "dali"))

