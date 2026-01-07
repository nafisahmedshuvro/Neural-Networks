"""
Preprocess and Combine Datasets
Processes GTZAN, Genius Lyrics, and DALI datasets for VAE training
"""
import os
import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path
import librosa
import soundfile as sf
from tqdm import tqdm

def process_gtzan_dataset(gtzan_dir, output_dir, target_sr=22050, duration=30):
    """
    Process GTZAN dataset: extract audio and genre labels
    
    Args:
        gtzan_dir: Directory containing GTZAN dataset
        output_dir: Output directory for processed files
        target_sr: Target sample rate
        duration: Segment duration in seconds
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "audio"), exist_ok=True)
    
    # GTZAN structure: genres_original/blues/blues.00000.wav
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    metadata = []
    
    print("Processing GTZAN dataset...")
    
    # Look for audio files
    audio_files = []
    for genre in genres:
        genre_dir = os.path.join(gtzan_dir, "genres_original", genre)
        if not os.path.exists(genre_dir):
            # Try alternative structure
            genre_dir = os.path.join(gtzan_dir, genre)
        
        if os.path.exists(genre_dir):
            for audio_file in os.listdir(genre_dir):
                if audio_file.endswith(('.wav', '.mp3', '.flac')):
                    audio_files.append((os.path.join(genre_dir, audio_file), genre))
    
    print(f"Found {len(audio_files)} audio files")
    
    for audio_path, genre in tqdm(audio_files, desc="Processing GTZAN"):
        try:
            # Load and preprocess audio
            y, sr = librosa.load(audio_path, sr=target_sr, duration=duration)
            y = librosa.util.normalize(y)
            
            # Pad or trim to fixed length
            target_length = target_sr * duration
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)), mode='constant')
            elif len(y) > target_length:
                y = y[:target_length]
            
            # Save processed audio
            filename = os.path.basename(audio_path)
            output_path = os.path.join(output_dir, "audio", filename)
            sf.write(output_path, y, target_sr)
            
            metadata.append({
                'file_name': filename,
                'genre': genre,
                'source': 'gtzan',
                'language': 'english'  # GTZAN is primarily English
            })
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue
    
    # Save metadata
    df_metadata = pd.DataFrame(metadata)
    df_metadata.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
    
    print(f"Processed {len(metadata)} files from GTZAN")
    return df_metadata

def process_genius_lyrics(genius_dir, output_dir):
    """
    Process Genius Lyrics dataset: extract lyrics and language information
    
    Args:
        genius_dir: Directory containing Genius dataset
        output_dir: Output directory for processed lyrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Processing Genius Lyrics dataset...")
    
    # Look for CSV or JSON files
    lyrics_files = []
    for ext in ['*.csv', '*.json', '*.jsonl']:
        lyrics_files.extend(Path(genius_dir).rglob(ext))
    
    if not lyrics_files:
        print(f"No lyrics files found in {genius_dir}")
        return None
    
    print(f"Found {len(lyrics_files)} lyrics files")
    
    all_lyrics = []
    
    for lyrics_file in lyrics_files:
        try:
            if lyrics_file.suffix == '.csv':
                df = pd.read_csv(lyrics_file)
                # Common column names: title, artist, lyrics, language, etc.
                if 'lyrics' in df.columns and 'language' in df.columns:
                    all_lyrics.append(df[['title', 'artist', 'lyrics', 'language']])
            elif lyrics_file.suffix in ['.json', '.jsonl']:
                with open(lyrics_file, 'r', encoding='utf-8') as f:
                    if lyrics_file.suffix == '.jsonl':
                        data = [json.loads(line) for line in f]
                    else:
                        data = json.load(f)
                    
                    # Convert to DataFrame
                    if isinstance(data, list):
                        df = pd.DataFrame(data)
                        if 'lyrics' in df.columns and 'language' in df.columns:
                            all_lyrics.append(df[['title', 'artist', 'lyrics', 'language']])
        except Exception as e:
            print(f"Error processing {lyrics_file}: {e}")
            continue
    
    if all_lyrics:
        df_lyrics = pd.concat(all_lyrics, ignore_index=True)
        
        # Filter for English and Bangla (or Bengali)
        df_lyrics = df_lyrics[
            df_lyrics['language'].str.lower().isin(['english', 'en', 'bangla', 'bengali', 'bn', 'bn-bd'])
        ]
        
        # Save processed lyrics
        df_lyrics.to_csv(os.path.join(output_dir, "lyrics_processed.csv"), index=False)
        
        print(f"Processed {len(df_lyrics)} lyrics entries")
        print(f"Languages: {df_lyrics['language'].value_counts().to_dict()}")
        
        return df_lyrics
    
    return None

def process_dali_dataset(dali_dir, output_dir, target_sr=22050):
    """
    Process DALI dataset: extract audio, lyrics, and annotations
    
    Args:
        dali_dir: Directory containing DALI dataset
        output_dir: Output directory for processed files
        target_sr: Target sample rate
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "audio"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)
    
    print("Processing DALI dataset...")
    print("Note: DALI requires the dali_code library")
    print("Install with: pip install dali-dataset")
    
    try:
        import dali_code
    except ImportError:
        print("Warning: dali_code not installed. Skipping DALI processing.")
        print("Install: pip install dali-dataset")
        return None
    
    # DALI structure: dali_data/...
    dali_data_path = os.path.join(dali_dir, "DALI", "dali_data")
    if not os.path.exists(dali_data_path):
        dali_data_path = os.path.join(dali_dir, "dali_data")
    
    if not os.path.exists(dali_data_path):
        print(f"DALI data directory not found at {dali_data_path}")
        return None
    
    metadata = []
    
    # Get list of DALI entries
    dali_info = dali_code.get_info(dali_data_path)
    
    print(f"Found {len(dali_info)} DALI entries")
    
    for dali_id, info in tqdm(list(dali_info.items())[:100], desc="Processing DALI"):  # Limit for demo
        try:
            # Get entry
            entry = dali_code.get_entry(dali_id, dali_data_path)
            
            # Extract lyrics
            if entry.annotations and 'annot' in entry.annotations:
                # Get text from annotations
                entry.horizontal2vertical()
                annot = entry.annotations['annot']['hierarchical']
                lyrics = dali_code.get_text(annot)
                lyrics_text = ' '.join(lyrics)
            else:
                lyrics_text = ""
            
            # Note: Audio needs to be downloaded separately
            # For now, we'll just extract metadata
            
            metadata.append({
                'dali_id': dali_id,
                'title': info.get('title', ''),
                'artist': info.get('artist', ''),
                'lyrics': lyrics_text,
                'source': 'dali',
                'language': 'mixed'  # DALI contains multiple languages
            })
            
        except Exception as e:
            print(f"Error processing DALI entry {dali_id}: {e}")
            continue
    
    df_dali = pd.DataFrame(metadata)
    df_dali.to_csv(os.path.join(output_dir, "dali_metadata.csv"), index=False)
    
    print(f"Processed {len(metadata)} DALI entries")
    return df_dali

def combine_datasets(gtzan_metadata, genius_lyrics, dali_metadata, output_dir):
    """
    Combine metadata from all datasets
    
    Args:
        gtzan_metadata: GTZAN metadata DataFrame
        genius_lyrics: Genius lyrics DataFrame
        dali_metadata: DALI metadata DataFrame
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nCombining datasets...")
    
    combined_metadata = []
    
    # Add GTZAN entries
    if gtzan_metadata is not None and len(gtzan_metadata) > 0:
        combined_metadata.append(gtzan_metadata)
    
    # Add Genius lyrics (will need to match with audio later)
    if genius_lyrics is not None and len(genius_lyrics) > 0:
        genius_df = genius_lyrics.copy()
        genius_df['source'] = 'genius'
        genius_df['file_name'] = genius_df['title'] + '_' + genius_df['artist']
        combined_metadata.append(genius_df[['file_name', 'title', 'artist', 'lyrics', 'language', 'source']])
    
    # Add DALI entries
    if dali_metadata is not None and len(dali_metadata) > 0:
        dali_df = dali_metadata.copy()
        dali_df['file_name'] = dali_df['dali_id']
        combined_metadata.append(dali_df[['file_name', 'title', 'artist', 'lyrics', 'language', 'source']])
    
    if combined_metadata:
        df_combined = pd.concat(combined_metadata, ignore_index=True)
        df_combined.to_csv(os.path.join(output_dir, "combined_metadata.csv"), index=False)
        
        print(f"\nCombined dataset statistics:")
        print(f"Total entries: {len(df_combined)}")
        print(f"By source: {df_combined['source'].value_counts().to_dict()}")
        if 'language' in df_combined.columns:
            print(f"By language: {df_combined['language'].value_counts().to_dict()}")
        
        return df_combined
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and combine datasets")
    parser.add_argument("--gtzan_dir", type=str, default="project/data/gtzan",
                       help="GTZAN dataset directory")
    parser.add_argument("--genius_dir", type=str, default="project/data/genius_lyrics",
                       help="Genius lyrics dataset directory")
    parser.add_argument("--dali_dir", type=str, default="project/data/dali",
                       help="DALI dataset directory")
    parser.add_argument("--output_dir", type=str, default="project/data/processed",
                       help="Output directory for processed data")
    parser.add_argument("--target_sr", type=int, default=22050, help="Target sample rate")
    parser.add_argument("--duration", type=int, default=30, help="Audio duration in seconds")
    
    args = parser.parse_args()
    
    # Process each dataset
    gtzan_metadata = None
    genius_lyrics = None
    dali_metadata = None
    
    # Process GTZAN
    if os.path.exists(args.gtzan_dir):
        gtzan_output = os.path.join(args.output_dir, "gtzan")
        gtzan_metadata = process_gtzan_dataset(args.gtzan_dir, gtzan_output, 
                                               args.target_sr, args.duration)
    else:
        print(f"GTZAN directory not found: {args.gtzan_dir}")
    
    # Process Genius Lyrics
    if os.path.exists(args.genius_dir):
        genius_output = os.path.join(args.output_dir, "genius")
        genius_lyrics = process_genius_lyrics(args.genius_dir, genius_output)
    else:
        print(f"Genius directory not found: {args.genius_dir}")
    
    # Process DALI
    if os.path.exists(args.dali_dir):
        dali_output = os.path.join(args.output_dir, "dali")
        dali_metadata = process_dali_dataset(args.dali_dir, dali_output, args.target_sr)
    else:
        print(f"DALI directory not found: {args.dali_dir}")
    
    # Combine datasets
    combined = combine_datasets(gtzan_metadata, genius_lyrics, dali_metadata, args.output_dir)
    
    print("\nPreprocessing complete!")
    print(f"Combined metadata saved to: {os.path.join(args.output_dir, 'combined_metadata.csv')}")



