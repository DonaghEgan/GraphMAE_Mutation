"""
Configuration module for Graph Neural Network Mutation Data analysis.
Contains all hyperparameters, paths, and settings in one centralized location.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    feats_in: int = 48
    feats_out: int = 128
    dropout: float = 0.5
    residual: bool = True
    mask_rate: float = 0.3
    replace_rate: float = 0.05
    loss_fn: str = 'sce'  # Options: 'mse', 'sce'
    alpha: int = 2
    
@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 500
    batch_size: int = 32
    learning_rate: float = 0.00015
    weight_decay: float = 2e-4
    early_stopping_patience: int = 50
    gradient_clip_norm: float = 1.0

    # Learning rate scheduler
    scheduler_T_max: int = 50
    scheduler_eta_min: float = 1e-6
    
@dataclass
class DataConfig:
    """Data configuration."""
    study_name: str = 'msk_pan_2017'
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 3

@dataclass
class PathConfig:
    """Path configuration."""
    project_root: str = "/home/degan/GraphMAE_Mutation"
    results_dir: str = "results"
    
    @property
    def results_path(self) -> str:
        return os.path.join(self.project_root, self.results_dir)
   
@dataclass
class Config:
    """Main configuration class combining all sub-configurations."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # Device configuration
    device_preference: str = "cuda"  # "cuda", "cpu", or "auto"
    
    # Logging configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_splits()
        self._create_directories()
    
    def _validate_splits(self):
        """Validate that data splits sum to 1.0."""
        total_split = self.data.train_split + self.data.val_split + self.data.test_split
        if not (0.99 <= total_split <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Data splits must sum to 1.0, got {total_split}")
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.paths.results_path,
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
   
    def get_results_filename(self, base_name: str, timestamp: str, extension: str = "csv") -> str:
        """Generate filename for results with embedding suffix."""
        return f"{base_name}_{timestamp}.{extension}"

def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or return default configuration.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Config: Configuration object
    """
    if config_path and os.path.exists(config_path):
        # TODO: Implement loading from JSON/YAML file if needed
        pass
    
    return Config()

# Default configuration instance
default_config = Config()