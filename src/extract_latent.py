"""
Extract Latent Features from Trained VAE
"""
import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from vae import BasicVAE, ConvVAE

class AudioDataset(Dataset):
    """Dataset for audio features"""
    def __init__(self, features_path, normalize=True):
        self.features = np.load(features_path)
        self.original_features = self.features.copy()
        
        if normalize:
            self.mean = np.mean(self.features, axis=0)
            self.std = np.std(self.features, axis=0) + 1e-8
            self.features = (self.features - self.mean) / self.std
        
        self.features = torch.FloatTensor(self.features)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]

def extract_latent_features(model, data_loader, device):
    """
    Extract latent features from VAE encoder
    
    Args:
        model: Trained VAE model
        data_loader: DataLoader for data
        device: torch device
    """
    model.eval()
    latent_features = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            
            # Add channel dimension for ConvVAE if needed
            if isinstance(model, ConvVAE) and batch.dim() == 2:
                batch = batch.view(batch.size(0), 1, 128, 128)
            
            mu, logvar = model.encode(batch)
            z = model.reparameterize(mu, logvar)
            
            latent_features.append(z.cpu().numpy())
    
    latent_features = np.concatenate(latent_features, axis=0)
    return latent_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract latent features from VAE")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained VAE model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to features .npy file")
    parser.add_argument("--model_type", type=str, choices=['basic', 'conv'], default='basic',
                       help="Type of VAE model")
    parser.add_argument("--latent_dim", type=int, default=32, help="Latent dimension")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for latent features")
    
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    dataset = AudioDataset(args.data_path)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Load model
    if args.model_type == 'basic':
        input_dim = dataset.features.shape[1]
        model = BasicVAE(input_dim=input_dim, latent_dim=args.latent_dim)
    elif args.model_type == 'conv':
        model = ConvVAE(latent_dim=args.latent_dim)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Model loaded from {args.model_path}")
    
    # Extract latent features
    latent_features = extract_latent_features(model, data_loader, device)
    
    # Save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.save(args.output_path, latent_features)
    
    print(f"Latent features extracted: {latent_features.shape}")
    print(f"Saved to: {args.output_path}")



