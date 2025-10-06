"""
Main training script for GraphMAE Mutation Analysis.

This script orchestrates the entire training pipeline including:
- Data loading and preprocessing
- Model setup and training
- Evaluation and results saving
"""

import os
import sys
import torch
import logging
import argparse
import json
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

# Import configurations and modules
from config import Config
from src.data_loader import DataLoader
from src.trainer import ModelTrainer

def setup_logging(config: Config):
    """
    Setup logging configuration.
    
    Args:
        config: Configuration object
    """
    # Create logs directory
    log_dir = os.path.join(config.paths.results_path, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format=config.log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


def setup_device(config: Config) -> torch.device:
    """
    Setup computation device based on configuration and availability.
    
    Args:
        config: Configuration object
        
    Returns:
        torch.device: Device to use for computation
    """
    if config.device_preference == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif config.device_preference == "cuda":
        if not torch.cuda.is_available():
            logging.warning("CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    logging.info(f"Using device: {device}")
    
    if device.type == "cuda":
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return device


def save_config(config: Config):
    """
    Save configuration to file for reproducibility.
    
    Args:
        config: Configuration object
    """
    config_dir = os.path.join(config.paths.results_path, "configs")
    os.makedirs(config_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_file = os.path.join(config_dir, f"config_{timestamp}.json")
    
    # Convert config dataclass to dictionary automatically
    config_dict = asdict(config)
    
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logging.info(f"Configuration saved to: {config_file}")

def create_data_loaders(data_loader: DataLoader, batch_size: int, device: torch.device):
    """
    Create PyTorch data loaders for train/val/test splits.
    
    Args:
        data_loader: DataLoader instance with loaded data
        batch_size: Batch size for data loaders
        device: Device to move data to
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
    
    # Create datasets for each split
    train_dataset = TensorDataset(
        data_loader.omics_tensor[data_loader.train_idx]
    )
    val_dataset = TensorDataset(
        data_loader.omics_tensor[data_loader.val_idx]
    )
    test_dataset = TensorDataset(
        data_loader.omics_tensor[data_loader.test_idx]
    )
    
    # Create data loaders
    train_loader = TorchDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    
    val_loader = TorchDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    
    test_loader = TorchDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    
    logging.info(f"Created data loaders with batch size: {batch_size}")
    logging.info(f"  Train batches: {len(train_loader)}")
    logging.info(f"  Val batches: {len(val_loader)}")
    logging.info(f"  Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


def save_training_results(results: dict, config: Config):
    """
    Save training results to file.
    
    Args:
        results: Dictionary containing training history
        config: Configuration object
    """
    results_dir = os.path.join(config.paths.results_path)
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"training_results_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Training results saved to: {results_file}")


def main(args):
    """
    Main training pipeline.
    
    Args:
        args: Command-line arguments
    """
    # Initialize configuration
    config = Config()
    
    # Override config with command-line arguments if provided
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.study:
        config.data.study_name = args.study
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("=" * 80)
    logger.info("GraphMAE Mutation Analysis - Training Pipeline")
    logger.info("=" * 80)
    
    # Save configuration
    save_config(config)
    
    # Setup device
    device = setup_device(config)
    
    try:
        # ====== DATA LOADING ======
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: DATA LOADING AND PREPROCESSING")
        logger.info("=" * 80)
        
        data_loader = DataLoader(config)
        
        # Load all data
        (omics_tensor, clin_tensor, osurv_tensor, 
         sample_embeddings_tensor, adj_matrix) = data_loader.load_all_data()
        
        # Get feature dimensions
        feature_dims = data_loader.get_feature_dimensions()
        feats_in = feature_dims['features_omics']
        
        logger.info(f"\nData loading completed:")
        logger.info(f"  Input features: {feats_in}")
        logger.info(f"  Number of genes: {adj_matrix.shape[0]}")
        logger.info(f"  Training samples: {len(data_loader.train_idx)}")
        logger.info(f"  Validation samples: {len(data_loader.val_idx)}")
        logger.info(f"  Test samples: {len(data_loader.test_idx)}")
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            data_loader, config.training.batch_size, device
        )
        
        # Move adjacency matrix to device
        adj_matrix = adj_matrix.to(device)
        
        # ====== MODEL SETUP ======
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: MODEL SETUP")
        logger.info("=" * 80)
        
        trainer = ModelTrainer(config, device)
        trainer.setup_model(feats_in)
        trainer.setup_optimizer_and_scheduler()
        
        # ====== TRAINING ======
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: MODEL TRAINING")
        logger.info("=" * 80)
        
        training_results = trainer.train(train_loader, val_loader, adj_matrix)
        
        # Save training results
        save_training_results(training_results, config)
        
        # ====== EVALUATION ======
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: TEST SET EVALUATION")
        logger.info("=" * 80)
        
        test_results = trainer.evaluate_test_set(test_loader, adj_matrix)
        
        logger.info(f"\nFinal Results:")
        logger.info(f"  Best Validation Loss: {training_results['best_val_loss']:.4f}")
        logger.info(f"  Test Loss: {test_results['test_loss']:.4f}")
        
        # Save test results
        test_results_file = os.path.join(
            config.paths.results_path,
            f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(test_results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        logger.info(f"\nTest results saved to: {test_results_file}")
        
        # ====== EMBEDDINGS EXTRACTION (Optional) ======
        if args.save_embeddings:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 5: EXTRACTING EMBEDDINGS")
            logger.info("=" * 80)
            
            train_embeddings = trainer.get_embeddings(train_loader, adj_matrix)
            
            embeddings_file = os.path.join(
                config.paths.results_path,
                f"embeddings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            )
            torch.save(train_embeddings, embeddings_file)
            logger.info(f"Embeddings saved to: {embeddings_file}")
        
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"\n{'=' * 80}")
        logger.error(f"ERROR: Training pipeline failed!")
        logger.error(f"{'=' * 80}")
        logger.error(f"Error details: {str(e)}", exc_info=True)
        sys.exit(1)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train GraphMAE model on mutation data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for training (overrides config)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)"
    )
    
    # Data arguments
    parser.add_argument(
        "--study",
        type=str,
        default=None,
        help="Study name to download and process (overrides config)"
    )
    
    # Output arguments
    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Save node embeddings after training"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
