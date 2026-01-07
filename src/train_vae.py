"""
VAE Training Script
"""
import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json

from vae import BasicVAE, ConvVAE, vae_loss

class AudioDataset(Dataset):
    """Dataset for audio features"""
    def __init__(self, features_path, normalize=True):
        self.features = np.load(features_path)
        
        if normalize:
            self.mean = np.mean(self.features, axis=0)
            self.std = np.std(self.features, axis=0) + 1e-8
            self.features = (self.features - self.mean) / self.std
        
        self.features = torch.FloatTensor(self.features)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]

def train_vae(model, train_loader, device, epochs=100, lr=1e-3, beta=1.0, save_path=None):
    """
    Train VAE model
    
    Args:
        model: VAE model
        train_loader: DataLoader for training data
        device: torch device
        epochs: Number of training epochs
        lr: Learning rate
        beta: Beta parameter for Beta-VAE
        save_path: Path to save model
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            batch = batch.to(device)
            
            # Add channel dimension for ConvVAE if needed
            if isinstance(model, ConvVAE) and batch.dim() == 2:
                # Assume input needs reshaping (for spectrograms)
                batch = batch.view(batch.size(0), 1, 128, 128)
            
            optimizer.zero_grad()
            
            recon_batch, mu, logvar, z = model(batch)
            
            # Flatten for loss calculation if needed
            if isinstance(model, ConvVAE):
                batch_flat = batch.view(batch.size(0), -1)
                recon_flat = recon_batch.view(recon_batch.size(0), -1)
            else:
                batch_flat = batch
                recon_flat = recon_batch
            
            loss, recon_loss, kl_loss = vae_loss(recon_flat, batch_flat, mu, logvar, beta=beta)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_recon = total_recon_loss / len(train_loader)
        avg_kl = total_kl_loss / len(train_loader)
        
        train_losses.append({
            'epoch': epoch + 1,
            'total_loss': avg_loss,
            'recon_loss': avg_recon,
            'kl_loss': avg_kl
        })
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, KL: {avg_kl:.4f})")
        
        # Save checkpoint
        if save_path and (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': train_losses
            }, save_path)
    
    # Final save
    if save_path:
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': train_losses
        }, save_path)
    
    return train_losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE")
    parser.add_argument("--data_path", type=str, required=True, help="Path to features .npy file")
    parser.add_argument("--model_type", type=str, choices=['basic', 'conv'], default='basic',
                       help="Type of VAE model")
    parser.add_argument("--latent_dim", type=int, default=32, help="Latent dimension")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta parameter for Beta-VAE")
    parser.add_argument("--save_path", type=str, default="project/models/vae_model.pth",
                       help="Path to save model")
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    dataset = AudioDataset(args.data_path)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Create model
    if args.model_type == 'basic':
        input_dim = dataset.features.shape[1]
        model = BasicVAE(input_dim=input_dim, latent_dim=args.latent_dim)
    elif args.model_type == 'conv':
        model = ConvVAE(latent_dim=args.latent_dim)
    
    print(f"Model created: {args.model_type} VAE with latent_dim={args.latent_dim}")
    print(f"Training on {len(dataset)} samples")
    
    # Create save directory
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # Train
    losses = train_vae(
        model, train_loader, device,
        epochs=args.epochs, lr=args.lr, beta=args.beta,
        save_path=args.save_path
    )
    
    print(f"Training complete! Model saved to {args.save_path}")



