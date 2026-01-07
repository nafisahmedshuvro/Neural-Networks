"""
Extract Lyrics Features
Extracts text embeddings from lyrics using Sentence-BERT or BERT
"""
import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

def extract_lyrics_embeddings(lyrics_df, model_name='all-MiniLM-L6-v2', output_path=None):
    """
    Extract embeddings from lyrics using Sentence-BERT
    
    Args:
        lyrics_df: DataFrame with 'lyrics' column
        model_name: Sentence-BERT model name
        output_path: Path to save embeddings
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Installing sentence-transformers...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'sentence-transformers'])
        from sentence_transformers import SentenceTransformer
    
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Filter out empty lyrics
    lyrics_df = lyrics_df[lyrics_df['lyrics'].notna() & (lyrics_df['lyrics'] != '')]
    
    print(f"Extracting embeddings for {len(lyrics_df)} lyrics...")
    
    # Extract embeddings
    lyrics_list = lyrics_df['lyrics'].tolist()
    embeddings = model.encode(lyrics_list, show_progress_bar=True, batch_size=32)
    
    # Save embeddings
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, embeddings)
        
        # Save mapping
        mapping_df = pd.DataFrame({
            'index': range(len(lyrics_df)),
            'file_name': lyrics_df['file_name'].values if 'file_name' in lyrics_df.columns else lyrics_df.index.values,
            'title': lyrics_df['title'].values if 'title' in lyrics_df.columns else [''] * len(lyrics_df),
            'language': lyrics_df['language'].values if 'language' in lyrics_df.columns else ['unknown'] * len(lyrics_df)
        })
        mapping_path = output_path.replace('.npy', '_mapping.csv')
        mapping_df.to_csv(mapping_path, index=False)
        
        print(f"Embeddings saved to: {output_path}")
        print(f"Mapping saved to: {mapping_path}")
    
    return embeddings, lyrics_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract lyrics embeddings")
    parser.add_argument("--lyrics_csv", type=str, required=True,
                       help="Path to CSV file with lyrics")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Output path for embeddings .npy file")
    parser.add_argument("--model_name", type=str, default='all-MiniLM-L6-v2',
                       help="Sentence-BERT model name")
    
    args = parser.parse_args()
    
    # Load lyrics
    df = pd.read_csv(args.lyrics_csv)
    
    if 'lyrics' not in df.columns:
        print("Error: 'lyrics' column not found in CSV")
        print(f"Available columns: {df.columns.tolist()}")
        exit(1)
    
    # Extract embeddings
    embeddings, lyrics_df = extract_lyrics_embeddings(df, args.model_name, args.output_path)
    
    print(f"Extracted embeddings shape: {embeddings.shape}")



