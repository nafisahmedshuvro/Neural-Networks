"""
Extract Hybrid Features
Combines audio and lyrics features for multi-modal VAE
"""
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def combine_audio_lyrics_features(audio_features_path, lyrics_features_path, 
                                  audio_mapping_path=None, lyrics_mapping_path=None,
                                  output_path=None, fusion_method='concatenate'):
    """
    Combine audio and lyrics features
    
    Args:
        audio_features_path: Path to audio features .npy file
        lyrics_features_path: Path to lyrics embeddings .npy file
        audio_mapping_path: Path to audio file mapping CSV
        lyrics_mapping_path: Path to lyrics mapping CSV
        output_path: Output path for combined features
        fusion_method: 'concatenate' or 'weighted_average'
    """
    # Load features
    audio_features = np.load(audio_features_path)
    lyrics_features = np.load(lyrics_features_path)
    
    print(f"Audio features shape: {audio_features.shape}")
    print(f"Lyrics features shape: {lyrics_features.shape}")
    
    # Load mappings if available
    audio_mapping = None
    lyrics_mapping = None
    
    if audio_mapping_path and os.path.exists(audio_mapping_path):
        audio_mapping = pd.read_csv(audio_mapping_path)
        print(f"Audio mapping: {len(audio_mapping)} entries")
    
    if lyrics_mapping_path and os.path.exists(lyrics_mapping_path):
        lyrics_mapping = pd.read_csv(lyrics_mapping_path)
        print(f"Lyrics mapping: {len(lyrics_mapping)} entries")
    
    # Match audio and lyrics by file name or index
    if audio_mapping is not None and lyrics_mapping is not None:
        # Try to match by file_name
        if 'file_name' in audio_mapping.columns and 'file_name' in lyrics_mapping.columns:
            merged = pd.merge(audio_mapping, lyrics_mapping, on='file_name', how='inner', suffixes=('_audio', '_lyrics'))
            
            if len(merged) == 0:
                print("Warning: No matches found by file_name. Using all available features separately.")
                # Use all features, pad if necessary
                min_samples = min(len(audio_features), len(lyrics_features))
                audio_features = audio_features[:min_samples]
                lyrics_features = lyrics_features[:min_samples]
            else:
                print(f"Matched {len(merged)} samples by file_name")
                audio_indices = merged['index_audio'].values
                lyrics_indices = merged['index_lyrics'].values
                audio_features = audio_features[audio_indices]
                lyrics_features = lyrics_features[lyrics_indices]
    
    # Normalize features
    scaler_audio = StandardScaler()
    scaler_lyrics = StandardScaler()
    
    audio_features_scaled = scaler_audio.fit_transform(audio_features)
    lyrics_features_scaled = scaler_lyrics.fit_transform(lyrics_features)
    
    # Combine features
    if fusion_method == 'concatenate':
        combined_features = np.concatenate([audio_features_scaled, lyrics_features_scaled], axis=1)
    elif fusion_method == 'weighted_average':
        # Weighted average (you can adjust weights)
        audio_weight = 0.6
        lyrics_weight = 0.4
        
        # Normalize to same dimension (use PCA or padding)
        min_dim = min(audio_features_scaled.shape[1], lyrics_features_scaled.shape[1])
        audio_reduced = audio_features_scaled[:, :min_dim]
        lyrics_reduced = lyrics_features_scaled[:, :min_dim]
        
        combined_features = audio_weight * audio_reduced + lyrics_weight * lyrics_reduced
    else:
        raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    print(f"Combined features shape: {combined_features.shape}")
    
    # Save
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, combined_features)
        print(f"Combined features saved to: {output_path}")
    
    return combined_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine audio and lyrics features")
    parser.add_argument("--audio_features", type=str, required=True,
                       help="Path to audio features .npy file")
    parser.add_argument("--lyrics_features", type=str, required=True,
                       help="Path to lyrics embeddings .npy file")
    parser.add_argument("--audio_mapping", type=str, default=None,
                       help="Path to audio mapping CSV (optional)")
    parser.add_argument("--lyrics_mapping", type=str, default=None,
                       help="Path to lyrics mapping CSV (optional)")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Output path for combined features")
    parser.add_argument("--fusion_method", type=str, choices=['concatenate', 'weighted_average'],
                       default='concatenate', help="Feature fusion method")
    
    args = parser.parse_args()
    
    combined = combine_audio_lyrics_features(
        args.audio_features,
        args.lyrics_features,
        args.audio_mapping,
        args.lyrics_mapping,
        args.output_path,
        args.fusion_method
    )



