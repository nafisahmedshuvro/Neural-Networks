"""
VAE Training Script - Fallback using scikit-learn Autoencoder
Use this if PyTorch DLL error persists
"""
import os
import argparse
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pickle
from tqdm import tqdm

def train_sklearn_autoencoder(features, latent_dim=32, hidden_dims=[64, 32], epochs=100):
    """
    Train a simple autoencoder using scikit-learn MLP
    This is a fallback when PyTorch doesn't work
    """
    print("Using scikit-learn Autoencoder (PyTorch fallback)")
    print(f"Input shape: {features.shape}")
    
    # Handle 2D/3D input (e.g., spectrograms)
    original_shape = features.shape
    if len(features.shape) > 2:
        # Flatten 2D/3D arrays (e.g., spectrograms: n_samples, height, width)
        print(f"Flattening {len(features.shape)}D input to 2D")
        features = features.reshape(features.shape[0], -1)
        print(f"Flattened shape: {features.shape}")
    
    # Adjust latent_dim if it's larger than input features
    max_latent = min(latent_dim, features.shape[1])
    if latent_dim > features.shape[1]:
        print(f"Warning: latent_dim ({latent_dim}) > input features ({features.shape[1]})")
        print(f"Using latent_dim = {max_latent} instead")
        latent_dim = max_latent
    
    print(f"Latent dimension: {latent_dim}")
    
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    print(f"Features scaled shape: {features_scaled.shape}")
    
    # Use PCA as encoder (simple and effective)
    from sklearn.decomposition import PCA
    
    print("Training encoder (PCA)...")
    encoder_pca = PCA(n_components=latent_dim, random_state=42)
    latent_features = encoder_pca.fit_transform(features_scaled)
    print(f"Latent features shape: {latent_features.shape}")
    
    # Decoder: MLP from latent to original
    print("Training decoder (MLP)...")
    decoder = MLPRegressor(
        hidden_layer_sizes=hidden_dims + [features.shape[1]],
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=min(32, len(features)),
        learning_rate='adaptive',
        max_iter=epochs,
        random_state=42,
        verbose=True,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    decoder.fit(latent_features, features_scaled)
    print("Training complete!")
    
    return encoder_pca, decoder, scaler, latent_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE using scikit-learn (fallback)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to features .npy file")
    parser.add_argument("--latent_dim", type=int, default=32, help="Latent dimension")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--save_path", type=str, default="project/models/vae_sklearn.pkl",
                       help="Path to save model")
    
    args = parser.parse_args()
    
    # Load features
    features = np.load(args.data_path)
    print(f"Loaded features: {features.shape}")
    
    # Train autoencoder
    encoder, decoder, scaler, latent_features = train_sklearn_autoencoder(
        features, latent_dim=args.latent_dim, epochs=args.epochs
    )
    
    # Save model
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    model_data = {
        'encoder': encoder,
        'decoder': decoder,
        'scaler': scaler,
        'latent_features': latent_features,
        'latent_dim': args.latent_dim
    }
    
    with open(args.save_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved to: {args.save_path}")
    print(f"Latent features shape: {latent_features.shape}")
    print("\nNote: This is a simplified autoencoder using scikit-learn.")
    print("For full VAE functionality, fix PyTorch installation.")

