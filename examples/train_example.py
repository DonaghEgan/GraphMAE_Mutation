"""
Example script showing how to run the GraphMAE training pipeline programmatically.
"""

import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config import Config
from src.data_loader import DataLoader
from src.trainer import ModelTrainer


def main():
    """Example training pipeline."""
    
    # 1. Initialize configuration
    config = Config()
    
    # Optional: Modify configuration
    config.training.epochs = 100
    config.training.batch_size = 32
    config.training.learning_rate = 0.001
    
    print("Configuration:")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Study: {config.data.study_name}")
    
    # 2. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # 3. Load data
    print("\nLoading data...")
    data_loader = DataLoader(config)
    (omics_tensor, clin_tensor, osurv_tensor, 
     sample_embeddings_tensor, adj_matrix) = data_loader.load_all_data()
    
    # Get feature dimensions
    feature_dims = data_loader.get_feature_dimensions()
    feats_in = feature_dims['features_omics']
    
    print(f"Data loaded successfully!")
    print(f"  Input features: {feats_in}")
    print(f"  Number of genes: {adj_matrix.shape[0]}")
    
    # 4. Create data loaders
    from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
    
    train_dataset = TensorDataset(omics_tensor[data_loader.train_idx])
    val_dataset = TensorDataset(omics_tensor[data_loader.val_idx])
    test_dataset = TensorDataset(omics_tensor[data_loader.test_idx])
    
    train_loader = TorchDataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    val_loader = TorchDataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)
    test_loader = TorchDataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)
    
    # Move adjacency matrix to device
    adj_matrix = adj_matrix.to(device)
    
    # 5. Setup model and trainer
    print("\nSetting up model...")
    trainer = ModelTrainer(config, device)
    trainer.setup_model(feats_in)
    trainer.setup_optimizer_and_scheduler()
    
    # 6. Train model
    print("\nStarting training...")
    training_results = trainer.train(train_loader, val_loader, adj_matrix)
    
    print(f"\nTraining completed!")
    print(f"  Best validation loss: {training_results['best_val_loss']:.4f}")
    
    # 7. Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate_test_set(test_loader, adj_matrix)
    
    print(f"Test results:")
    print(f"  Test loss: {test_results['test_loss']:.4f}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
