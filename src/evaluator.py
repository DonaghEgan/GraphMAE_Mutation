"""
Evaluation module for Graph Neural Network Mutation Data analysis.
Handles model evaluation, metrics calculation, and results saving.
"""

import torch
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional
import cox_loss as cl
import model as m
from config import Config
from utils import MetricsTracker, FileManager, safe_float_conversion

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Handles model evaluation and metrics calculation."""
    
    def __init__(self, config: Config, device: torch.device):
        """
        Initialize ModelEvaluator.
        
        Args:
            config: Configuration object
            device: Device for computation
        """
        self.config = config
        self.device = device
        self.file_manager = FileManager(config)
    
    def evaluate_model(self, model: torch.nn.Module, data_loader, 
                      adj_matrix: torch.Tensor) -> Tuple[float, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            model: Model to evaluate
            data_loader: Data loader for evaluation
            adj_matrix: Adjacency matrix
            
        Returns:
            Tuple[float, float]: (average_loss, average_c_index)
        """

        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        try:
            with torch.no_grad():
                for batch in data_loader:
                    # Move batch to device
                    batch = batch.to(self.device)
                    num_batches += 1
                    
                    # Forward pass
                    loss, _ = model(batch.omics, adj_matrix, batch.clin, batch.sample_meta)
                    total_loss += loss.item()
                              
            if num_batches == 0:
                logger.warning("No batches found in data loader")
                return 0.0, 0.0
            
            avg_loss = total_loss / num_batches
            
            return avg_loss
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise
    
    def evaluate_on_test_set(self, model: torch.nn.Module, 
                           test_data_loader, adj_matrix: torch.Tensor,
                           model_paths: Dict[str, str]) -> Dict[str, float]:
        """
        Evaluate model on test set using different checkpoints.
        
        Args:
            model: Model to evaluate
            test_data_loader: Test data loader
            adj_matrix: Adjacency matrix
            model_paths: Dictionary of model checkpoint paths
            
        Returns:
            Dict[str, float]: Test results for different checkpoints
        """
        logger.info("Evaluating on test set...")
        results = {}
        
        for checkpoint_name, model_path in model_paths.items():
            try:
                # Load checkpoint
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Loaded checkpoint: {checkpoint_name}")
                
                # Evaluate
                test_loss = self.evaluate_model(model, test_data_loader, adj_matrix)
                
                results[f'test_loss_{checkpoint_name}'] = test_loss
                
                logger.info(f"Test results ({checkpoint_name}): Loss = {test_loss:.4f}")
                
            except FileNotFoundError:
                logger.warning(f"Checkpoint not found: {model_path}")
            except Exception as e:
                logger.error(f"Error evaluating checkpoint {checkpoint_name}: {e}")
        
        return results

class ResultsSaver:
    """Handles saving of training results and model checkpoints."""
    
    def __init__(self, config: Config):
        """
        Initialize ResultsSaver.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.file_manager = FileManager(config)
    
    def save_training_results(self, metrics_tracker: MetricsTracker) -> str:
        """
        Save training results to CSV file.
        
        Args:
            metrics_tracker: MetricsTracker containing training metrics
            
        Returns:
            str: Path to saved CSV file
        """
        logger.info("Saving training results...")
        
        try:
            # Create DataFrame
            results_df = pd.DataFrame({
                'epoch': list(range(len(metrics_tracker.train_losses))),
                'train_loss': metrics_tracker.train_losses,
                'val_loss': metrics_tracker.val_losses,
            })
            
            # Get filename and save
            filename = self.file_manager.get_results_filename("training_results")
            filepath = self.file_manager.get_results_path(filename)
            
            results_df.to_csv(filepath, index=False)
            
            logger.info(f"Training results saved to: {filepath}")
            logger.info(f"Columns: {list(results_df.columns)}")
            logger.info(f"Total epochs: {len(results_df)}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving training results: {e}")
            raise
    
    def save_training_summary(self, metrics_tracker: MetricsTracker, 
                            test_results: Optional[Dict[str, float]] = None) -> str:
        """
        Save training summary to CSV file.
        
        Args:
            metrics_tracker: MetricsTracker containing training metrics
            test_results: Optional test results dictionary
            
        Returns:
            str: Path to saved summary file
        """
        logger.info("Saving training summary...")
        
        try:
            # Get summary metrics
            summary_metrics = metrics_tracker.get_summary_metrics()
            
            # Prepare summary data
            metrics = ['best_val_loss', 'final_train_loss', 
                      'final_val_loss', 'total_epochs']
            
            values = [
                summary_metrics['best_val_loss'],
                summary_metrics['final_train_loss'],
                summary_metrics['final_val_loss'],
                summary_metrics['total_epochs'],
            ]
            
            # Add test results if available
            if test_results:
                for key, value in test_results.items():
                    metrics.append(key)
                    values.append(value)
            
            # Create DataFrame
            summary_df = pd.DataFrame({
                'metric': metrics,
                'value': values
            })
            
            # Get filename and save
            filename = self.file_manager.get_results_filename("training_summary")
            filepath = self.file_manager.get_results_path(filename)
            
            summary_df.to_csv(filepath, index=False)
            
            logger.info(f"Training summary saved to: {filepath}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving training summary: {e}")
            raise
    
    def save_model_checkpoint(self, model: torch.nn.Module, 
                            checkpoint_name: str) -> str:
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            checkpoint_name: Name for the checkpoint
            
        Returns:
            str: Path to saved model
        """
        try:
            filename = self.file_manager.get_model_filename(checkpoint_name)
            filepath = self.file_manager.get_model_path(filename)
            
            torch.save(model.state_dict(), filepath)
            logger.info(f"Model checkpoint saved: {filepath}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving model checkpoint: {e}")
            raise
    
    def get_model_paths(self) -> Dict[str, str]:
        """
        Get paths to saved model checkpoints.
        
        Returns:
            Dict[str, str]: Dictionary mapping checkpoint names to paths
        """
        early_stopping_model = self.file_manager.get_model_path(
            self.file_manager.get_model_filename("best_model"))
        
        best_ci_model = self.file_manager.get_model_path(
            self.file_manager.get_model_filename("best_cindex_model"))
        
        return {
            'early_stopping': early_stopping_model,
            'best_ci': best_ci_model
        }
