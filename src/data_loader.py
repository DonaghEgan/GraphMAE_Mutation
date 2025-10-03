"""
Data loading module for Graph Neural Network Mutation Data analysis.
Handles all data processing, loading, and preparation.
"""

import torch
import numpy as np
import random
import logging
from typing import Tuple, Dict, Any, List
import gc

import process_data as prc
import utility_functions as uf
import download_study as ds
import read_specific as rs
from config import Config


logger = logging.getLogger(__name__)


class DataLoader:

    """Handles data loading and preprocessing for the GNN mutation analysis."""
    
    def __init__(self, config: Config):
        """
        Initialize DataLoader with configuration.
        
        Args:
            config: Configuration object containing data settings
        """
        self.config = config
        self.data_dict = None
        self.gene_index = None
        self.sample_index = None
        self.adj_matrix = None
        
        # Processed tensors
        self.omics_tensor = None
        self.clin_tensor = None
        self.osurv_tensor = None
        self.sample_embeddings_tensor = None
        
        # Data splits
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None
    
    def download_and_process_data(self) -> None:
        """Download study data and process it."""
        logger.info(f"Downloading study: {self.config.data.study_name}")
        
        try:
            path, sources, urls = ds.download_study(name=self.config.data.study_name)
            logger.info(f"Data downloaded to: {path[0]}")
            
            # Process data files
            logger.info("Processing data files...")
            self.data_dict = prc.read_files(path[0])
            
            self.gene_index = self.data_dict['gene_index']
            self.sample_index = self.data_dict['sample_index']
            
            logger.info(f"Processed {len(self.sample_index)} samples and {len(self.gene_index)} genes")
            
        except Exception as e:
            logger.error(f"Error downloading/processing data: {e}")
            raise
    
    def extract_and_merge_features(self) -> None:
        """Extract features from data dictionary and merge them."""
        logger.info("Extracting and merging features...")
        
        try:
            # Extract mutation data
            protein_pos = self.data_dict['mutation']['protein_pos']
            var_type = self.data_dict['mutation']['variant_type_np']
            aa_sub = self.data_dict['mutation']['amino_acid']
            chrom_mut = self.data_dict['mutation']['chromosome_np']
            var_class_mut = self.data_dict['mutation']['var_class_np']
            fs_mut = self.data_dict['mutation']['frameshift']
            
            # Log shapes for verification
            logger.debug(f"Mutation features shapes:")
            logger.debug(f"  protein_pos: {protein_pos.shape}")
            logger.debug(f"  var_class_mut: {var_class_mut.shape}")
            logger.debug(f"  chrom_mut: {chrom_mut.shape}")
            logger.debug(f"  var_type: {var_type.shape}")
            logger.debug(f"  aa_sub: {aa_sub.shape}")
            logger.debug(f"  fs_mut: {fs_mut.shape}")
            
            # Extract SV data
            chrom_sv = self.data_dict['sv']['chromosome']
            var_class_sv = self.data_dict['sv']['var_class']
            region_sites = self.data_dict['sv']['region_sites']
            connection_type = self.data_dict['sv']['connection_type']
            sv_length = self.data_dict['sv']['sv_length']
            
            logger.debug(f"SV features shapes:")
            logger.debug(f"  chrom_sv: {chrom_sv.shape}")
            logger.debug(f"  var_class_sv: {var_class_sv.shape}")
            logger.debug(f"  region_sites: {region_sites.shape}")
            logger.debug(f"  connection_type: {connection_type.shape}")
            logger.debug(f"  sv_length: {sv_length.shape}")
            
            # Extract CNA data
            cna = self.data_dict['cna']['cna']
            cna = np.expand_dims(cna, axis=-1)
            logger.debug(f"CNA shape: {cna.shape}")
            
            # Extract clinical and sample data
            osurv_data = self.data_dict['os_array']
            clinical_data = self.data_dict['patient']
            sample_embeddings = self.data_dict['sample_meta']['embeddings']
            
            logger.debug(f"Clinical data shapes:")
            logger.debug(f"  osurv_data: {osurv_data.shape}")
            logger.debug(f"  clinical_data: {clinical_data.shape}")
            logger.debug(f"  sample_embeddings: {sample_embeddings.shape}")
            logger.debug(f"NaN values in survival data: {np.isnan(osurv_data).sum()}")
            
            # Merge features using utility functions
            var_class_mut_flat = uf.merge_last_two_dims(var_class_mut)
            chrom_mut_flat = uf.merge_last_two_dims(chrom_mut)
            aa_sub_flat = uf.merge_last_two_dims(aa_sub)
            var_type_flat = uf.merge_last_two_dims(var_type)
            
            chrom_sv_flat = uf.merge_last_two_dims(chrom_sv)
            var_class_sv_flat = uf.merge_last_two_dims(var_class_sv)
            region_sites_flat = uf.merge_last_two_dims(region_sites)
            
            # Create feature concatenation list
            arrays_to_concat = [
                protein_pos,
                fs_mut,
                var_class_mut_flat,
                chrom_mut_flat,
                var_type_flat,
                chrom_sv_flat,
                aa_sub_flat,
                var_class_sv_flat,
                region_sites_flat,
                sv_length,
                connection_type,
                cna
            ]
            
            # Concatenate all omics features
            uf.log_memory('Before feature concatenation')
            omics = np.concatenate(arrays_to_concat, axis=2)
            uf.log_memory('After feature concatenation')
            
            # Convert to tensors
            self.omics_tensor = torch.tensor(omics, dtype=torch.float32)
            self.clin_tensor = torch.tensor(clinical_data, dtype=torch.float32)
            self.osurv_tensor = torch.tensor(osurv_data, dtype=torch.float32)
            self.sample_embeddings_tensor = torch.tensor(sample_embeddings, dtype=torch.float32)
            
            logger.info(f"Created tensors:")
            logger.info(f"  omics: {self.omics_tensor.shape}")
            logger.info(f"  clinical: {self.clin_tensor.shape}")
            logger.info(f"  survival: {self.osurv_tensor.shape}")
            logger.info(f"  embeddings: {self.sample_embeddings_tensor.shape}")
            
            # Clean up memory
            del (arrays_to_concat, protein_pos, var_type_flat, aa_sub_flat, 
                 chrom_sv_flat, var_class_mut_flat, omics, clinical_data, 
                 osurv_data, chrom_mut_flat, fs_mut, sample_embeddings, cna)
            gc.collect()
            
            logger.info("Feature extraction and merging completed successfully")
            
        except Exception as e:
            logger.error(f"Error in feature extraction and merging: {e}")
            raise
    
    def create_train_val_test_splits(self) -> None:
        """Create train, validation, and test splits."""
        logger.info("Creating train/validation/test splits...")
        
        try:
            uf.log_memory('Before train-test split')
            
            # Set random seed for reproducibility
            random.seed(self.config.data.random_seed)
            
            # Get sample indices and shuffle
            sample_idx = list(self.sample_index.values())
            random.shuffle(sample_idx)
            
            # Calculate split sizes
            n_samples = len(self.sample_index)
            n_train = int(self.config.data.train_split * n_samples)
            n_val = int(self.config.data.val_split * n_samples)
            
            # Create splits
            self.train_idx = sample_idx[:n_train]
            self.val_idx = sample_idx[n_train:n_train + n_val]
            self.test_idx = sample_idx[n_train + n_val:]
            
            logger.info(f"Data splits created:")
            logger.info(f"  Training: {len(self.train_idx)} samples ({len(self.train_idx)/n_samples:.1%})")
            logger.info(f"  Validation: {len(self.val_idx)} samples ({len(self.val_idx)/n_samples:.1%})")
            logger.info(f"  Test: {len(self.test_idx)} samples ({len(self.test_idx)/n_samples:.1%})")
            
            uf.log_memory('After train-test split')
            
        except Exception as e:
            logger.error(f"Error creating data splits: {e}")
            raise
    
    def load_adjacency_matrix(self) -> torch.Tensor:
        """Load and return the adjacency matrix."""
        logger.info("Loading adjacency matrix...")
        
        try:
            uf.log_memory('Before adjacency matrix creation')
            
            self.adj_matrix = rs.read_reactome_new(gene_index=self.gene_index)
            
            if self.adj_matrix is None:
                raise ValueError("Adjacency matrix was not returned correctly")
            
            # Convert to tensor if it's not already
            if not isinstance(self.adj_matrix, torch.Tensor):
                self.adj_matrix = torch.tensor(self.adj_matrix, dtype=torch.float32)
            
            row_sums = self.adj_matrix.sum(axis=1).mean()
            logger.info(f"Adjacency matrix shape: {self.adj_matrix.shape}")
            logger.info(f"Average number of neighbors per gene: {row_sums:.2f}")
            
            uf.log_memory('After adjacency matrix creation')
            
            return self.adj_matrix
            
        except Exception as e:
            logger.error(f"Error loading adjacency matrix: {e}")
            raise
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get the dimensions of different feature types."""
        if self.omics_tensor is None or self.clin_tensor is None or self.sample_embeddings_tensor is None:
            raise ValueError("Tensors not created yet. Call extract_and_merge_features() first.")
        
        return {
            'features_omics': self.omics_tensor.shape[2],
            'features_clin': self.clin_tensor.shape[1],
            'embedding_dim_string': self.sample_embeddings_tensor.shape[1],
            'max_tokens': len(self.gene_index)
        }
    
    def cleanup_memory(self) -> None:
        """Clean up large data structures to free memory."""
        logger.info("Cleaning up data dictionary to free memory...")
        if self.data_dict is not None:
            del self.data_dict
            self.data_dict = None
            gc.collect()
    
    def load_all_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Complete data loading pipeline.
        
        Returns:
            Tuple of (omics_tensor, clin_tensor, osurv_tensor, sample_embeddings_tensor, adj_matrix)
        """
        logger.info("Starting complete data loading pipeline...")
        
        try:
            # Download and process data
            self.download_and_process_data()
            
            # Extract and merge features
            self.extract_and_merge_features()
            
            # Create data splits
            self.create_train_val_test_splits()
            
            # Load adjacency matrix
            self.load_adjacency_matrix()
            
            # Clean up memory
            self.cleanup_memory()
            
            logger.info("Data loading pipeline completed successfully")
            
            return (self.omics_tensor, self.clin_tensor, self.osurv_tensor, 
                   self.sample_embeddings_tensor, self.adj_matrix)
            
        except Exception as e:
            logger.error(f"Error in data loading pipeline: {e}")
            raise


def load_data(config: Config) -> Tuple[DataLoader, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convenience function to load all data using the DataLoader.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (data_loader, omics_tensor, clin_tensor, osurv_tensor, sample_embeddings_tensor, adj_matrix)
    """
    data_loader = DataLoader(config)
    tensors = data_loader.load_all_data()
    
    return data_loader, *tensors