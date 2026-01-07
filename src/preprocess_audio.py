"""
Audio Preprocessing Script
Preprocesses raw audio files: resampling, segmentation, normalization
"""
import os
import argparse
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

def preprocess_audio(input_dir, output_dir, target_sr=22050, duration=30):
    """
    Preprocess audio files: resample, segment, normalize
    
    Args:
        input_dir: Directory containing raw audio files
        output_dir: Directory to save processed files
        target_sr: Target sample rate (default: 22050 Hz)
        duration: Segment duration in seconds (default: 30s)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    audio_files = [f for f in os.listdir(input_dir) 
                   if f.endswith(('.wav', '.mp3', '.flac', '.m4a'))]
    
    print(f"Found {len(audio_files)} audio files")
    
    for audio_file in tqdm(audio_files, desc="Preprocessing audio"):
        try:
            # Load audio
            file_path = os.path.join(input_dir, audio_file)
            y, sr = librosa.load(file_path, sr=target_sr, duration=duration)
            
            # Normalize audio
            y = librosa.util.normalize(y)
            
            # If audio is shorter than duration, pad with zeros
            target_length = target_sr * duration
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)), mode='constant')
            elif len(y) > target_length:
                y = y[:target_length]
            
            # Save processed audio
            output_path = os.path.join(output_dir, audio_file.replace('.mp3', '.wav').replace('.m4a', '.wav'))
            sf.write(output_path, y, target_sr)
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue
    
    print(f"Preprocessing complete. Processed files saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess audio files")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory with raw audio")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed audio")
    parser.add_argument("--target_sr", type=int, default=22050, help="Target sample rate")
    parser.add_argument("--duration", type=int, default=30, help="Segment duration in seconds")
    
    args = parser.parse_args()
    preprocess_audio(args.input_dir, args.output_dir, args.target_sr, args.duration)



