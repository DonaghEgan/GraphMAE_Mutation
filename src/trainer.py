"""
Training module for Graph Neural Network Mutation Data analysis.
Contains training logic, optimizers, and schedulers.
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
from typing import Tuple, Optional, Dict
from tqdm import tqdm
from .model import GraphMae
from config import Config

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training with configurable loss functions and optimization."""
    
    def __init__(self, config: Config, device: torch.device):
        """
        Initialize ModelTrainer.
        
        Args:
            config: Configuration object
            device: Device for computation
        """
        self.config = config
        self.device = device
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def setup_model(self, feats_in: int) -> nn.Module:
        """
        Setup and initialize the GraphMAE model.
        
        Args:
            feats_in: Number of input features
            
        Returns:
            nn.Module: Initialized model
        """
        logger.info("Setting up GraphMAE model...")
        
        try:
            self.model = GraphMae(
                feats_in=feats_in,
                feats_out=self.config.model.feats_out,
                mask_rate=self.config.model.mask_rate,
                replace_rate=self.config.model.replace_rate,
                loss_fn=self.config.model.loss_fn,
                alpha=self.config.model.alpha
            )
            
            # Move model to device
            self.model = self.model.to(self.device)
            
            logger.info(f"Model initialized with:")
            logger.info(f"  Input features: {feats_in}")
            logger.info(f"  Output features: {self.config.model.feats_out}")
            logger.info(f"  Mask rate: {self.config.model.mask_rate}")
            logger.info(f"  Replace rate: {self.config.model.replace_rate}")
            logger.info(f"  Loss function: {self.config.model.loss_fn}")
            logger.info(f"  Alpha: {self.config.model.alpha}")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Error setting up model: {e}")
            raise
    
    def setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler."""
        logger.info("Setting up optimizer and scheduler...")
        
        try:
            # Setup optimizer
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=(0.9, 0.999)
            )
            
            # Setup scheduler
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.scheduler_T_max,
                eta_min=self.config.training.scheduler_eta_min
            )
            
            logger.info(f"Optimizer: AdamW (lr={self.config.training.learning_rate}, "
                       f"weight_decay={self.config.training.weight_decay})")
            logger.info(f"Scheduler: CosineAnnealingLR (T_max={self.config.training.scheduler_T_max}, "
                       f"eta_min={self.config.training.scheduler_eta_min})")
            
        except Exception as e:
            logger.error(f"Error setting up optimizer and scheduler: {e}")
            raise

    
    def train_epoch(self, train_data_loader, adj_matrix: torch.Tensor, 
                   epoch: int) -> float:
        """
        Train model for one epoch using GraphMAE.
        
        Args:
            train_data_loader: Training data loader
            adj_matrix: Adjacency matrix [G, G]
            epoch: Current epoch number
            
        Returns:
            float: Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        try:
            adj_matrix = adj_matrix.to(self.device)  # ✅ Move once
            for batch in train_data_loader:
                # Move batch to device
                # Unpack batch tuple from TensorDataset
                if isinstance(batch, (tuple, list)):
                    x = batch[0].to(self.device)
                else:
                    x = batch.to(self.device)
                num_batches += 1
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass - model returns loss directly
                loss, loss_dict = self.model(adj_matrix, x)
                
                # Check for invalid loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Invalid loss at batch {num_batches}, skipping...")
                    num_batches -= 1
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.training.gradient_clip_norm
                )
                
                # Optimizer step
                self.optimizer.step()
                
                # Accumulate metrics
                total_loss += loss.item()
            
            if num_batches == 0:
                logger.warning("No valid batches in training data")
                return 0.0
            
            avg_loss = total_loss / num_batches
            
            return avg_loss
            
        except Exception as e:
            logger.error(f"Error during training epoch {epoch}: {e}")
            raise
    
    def validate_epoch(self, val_data_loader, adj_matrix: torch.Tensor) -> float:
        """
        Validate model for one epoch.
        
        Args:
            val_data_loader: Validation data loader
            adj_matrix: Adjacency matrix [G, G]
            
        Returns:
            float: Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        try:
            with torch.no_grad():
                adj_matrix = adj_matrix.to(self.device)  # Move once
                for batch in val_data_loader:
                    # Move batch to device
                    # Unpack batch tuple from TensorDataset
                    if isinstance(batch, (tuple, list)):
                        x = batch[0].to(self.device)
                    else:
                        x = batch.to(self.device)
                    num_batches += 1
                    
                    # Forward pass
                    loss, loss_dict = self.model(adj_matrix, x)
                    
                    # Accumulate metrics
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        total_loss += loss.item()
                    else:
                        num_batches -= 1
            
            if num_batches == 0:
                logger.warning("No valid batches in validation data")
                return 0.0
            
            avg_loss = total_loss / num_batches
            
            return avg_loss
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            raise
    
    def train(self, train_data_loader, val_data_loader, adj_matrix: torch.Tensor) -> Dict[str, list]:
        """
        Main training loop for GraphMAE.
        
        Args:
            train_data_loader: Training data loader
            val_data_loader: Validation data loader
            adj_matrix: Adjacency matrix [G, G]
            
        Returns:
            Dict: Training history with train and validation losses
        """
        logger.info("Starting training...")
        logger.info(f"Training configuration:")
        logger.info(f"  Epochs: {self.config.training.epochs}")
        logger.info(f"  Batch size: {self.config.training.batch_size}")
        logger.info(f"  Learning rate: {self.config.training.learning_rate}")
        logger.info(f"  Early stopping patience: {self.config.training.early_stopping_patience}")
        
        try:
            # Training loop
            for epoch in tqdm(range(1, self.config.training.epochs + 1), 
                             desc="Training"):

                # Training step
                train_loss = self.train_epoch(train_data_loader, adj_matrix, epoch)
                
                # Validation step
                val_loss = self.validate_epoch(val_data_loader, adj_matrix)
                
                # Store losses
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                
                # Learning rate scheduler step
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                
                # Print progress
                if epoch % 5 == 0 or epoch == 1:
                    logger.info(f"Epoch {epoch}/{self.config.training.epochs} - "
                              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                              f"LR: {current_lr:.6f}")
                
                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    # Save best model
                    self.save_checkpoint("best_model.pt")
                    logger.info(f"New best validation loss: {val_loss:.4f}")
                else:
                    self.patience_counter += 1
                    
                if self.patience_counter >= self.config.training.early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}!")
                    break
            
            logger.info("Training completed successfully!")
            logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
            
            return {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'best_val_loss': self.best_val_loss
            }
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def evaluate_test_set(self, test_data_loader, adj_matrix: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            test_data_loader: Test data loader
            adj_matrix: Adjacency matrix [G, G]
            
        Returns:
            dict: Test evaluation results
        """
        logger.info("Evaluating on test set...")
        
        try:
            # Load best model
            self.load_checkpoint("best_model.pt")
            
            # Evaluate
            test_loss = self.validate_epoch(test_data_loader, adj_matrix)
            
            logger.info(f"Test loss: {test_loss:.4f}")
            
            return {'test_loss': test_loss}
            
        except Exception as e:
            logger.error(f"Error during test evaluation: {e}")
            raise
    
    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint.
        
        Args:
            filename: Name of checkpoint file
        """
        import os
        
        checkpoint_path = os.path.join(self.config.paths.results_path, filename)
        os.makedirs(self.config.paths.results_path, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, filename: str):
        """
        Load model checkpoint.
        
        Args:
            filename: Name of checkpoint file
        """
        import os
        
        checkpoint_path = os.path.join(self.config.paths.results_path, filename)
        
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint {checkpoint_path} not found!")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def get_embeddings(self, data_loader, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings from the trained model.
        
        Args:
            data_loader: Data loader
            adj_matrix: Adjacency matrix [G, G]
            
        Returns:
            torch.Tensor: Node embeddings
        """
        self.model.eval()
        embeddings_list = []
        
        try:
            with torch.no_grad():
                for batch in data_loader:
                    # Unpack batch tuple from TensorDataset
                    if isinstance(batch, (tuple, list)):
                        x = batch[0].to(self.device)
                    else:
                        x = batch.to(self.device)
                    adj_matrix = adj_matrix.to(self.device)
                    
                    # Get embeddings using the embed method
                    h = self.model.embed(adj_matrix, x)
                    embeddings_list.append(h.cpu())
            
            # Concatenate all embeddings
            all_embeddings = torch.cat(embeddings_list, dim=0)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            raise


def create_trainer(config: Config, device: torch.device, feats_in: int) -> ModelTrainer:
    """
    Create and setup a ModelTrainer with GraphMAE model.
    
    Args:
        config: Configuration object
        device: Device for computation
        feats_in: Number of input features
        
    Returns:
        ModelTrainer: Configured trainer with initialized model
    """
    trainer = ModelTrainer(config, device)
    trainer.setup_model(feats_in)
    trainer.setup_optimizer_and_scheduler()
    return trainer