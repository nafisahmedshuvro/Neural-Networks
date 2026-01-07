"""
Feature Extraction Script
Extracts MFCC, Mel-spectrogram, or other audio features
"""
import os
import argparse
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

def extract_mfcc(audio_dir, n_mfcc=13, hop_length=512):
    """
    Extract MFCC features from audio files
    
    Args:
        audio_dir: Directory containing audio files
        n_mfcc: Number of MFCC coefficients
        hop_length: Hop length for STFT
    """
    features = []
    file_names = []
    
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    
    for audio_file in tqdm(audio_files, desc="Extracting MFCC"):
        try:
            file_path = os.path.join(audio_dir, audio_file)
            y, sr = librosa.load(file_path, sr=22050)
            
            # Extract MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
            
            # Take mean across time dimension (or use other aggregation)
            mfcc_mean = np.mean(mfcc, axis=1)
            
            features.append(mfcc_mean)
            file_names.append(audio_file)
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue
    
    features = np.array(features)
    return features, file_names

def extract_melspectrogram(audio_dir, n_mels=128, hop_length=512):
    """
    Extract Mel-spectrogram features from audio files
    
    Args:
        audio_dir: Directory containing audio files
        n_mels: Number of mel filter banks
        hop_length: Hop length for STFT
    """
    features = []
    file_names = []
    
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    
    for audio_file in tqdm(audio_files, desc="Extracting Mel-spectrogram"):
        try:
            file_path = os.path.join(audio_dir, audio_file)
            y, sr = librosa.load(file_path, sr=22050)
            
            # Extract Mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Resize to fixed size (e.g., 128x128) for CNN input
            from scipy.ndimage import zoom
            target_shape = (128, 128)
            zoom_factors = (target_shape[0] / mel_spec_db.shape[0], 
                          target_shape[1] / mel_spec_db.shape[1])
            mel_spec_resized = zoom(mel_spec_db, zoom_factors, order=1)
            
            features.append(mel_spec_resized)
            file_names.append(audio_file)
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue
    
    features = np.array(features)
    return features, file_names

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract audio features")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory with audio files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for features")
    parser.add_argument("--feature_type", type=str, choices=['mfcc', 'melspectrogram'], 
                       default='mfcc', help="Type of features to extract")
    parser.add_argument("--n_mfcc", type=int, default=13, help="Number of MFCC coefficients")
    parser.add_argument("--n_mels", type=int, default=128, help="Number of mel filter banks")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.feature_type == 'mfcc':
        features, file_names = extract_mfcc(args.audio_dir, n_mfcc=args.n_mfcc)
        output_path = os.path.join(args.output_dir, 'mfcc_features.npy')
        np.save(output_path, features)
        
        # Save file names mapping
        df = pd.DataFrame({'file_name': file_names, 'index': range(len(file_names))})
        df.to_csv(os.path.join(args.output_dir, 'file_names.csv'), index=False)
        
    elif args.feature_type == 'melspectrogram':
        features, file_names = extract_melspectrogram(args.audio_dir, n_mels=args.n_mels)
        output_path = os.path.join(args.output_dir, 'melspectrogram_features.npy')
        np.save(output_path, features)
        
        # Save file names mapping
        df = pd.DataFrame({'file_name': file_names, 'index': range(len(file_names))})
        df.to_csv(os.path.join(args.output_dir, 'file_names.csv'), index=False)
    
    print(f"Features extracted: {features.shape}")
    print(f"Saved to: {output_path}")



