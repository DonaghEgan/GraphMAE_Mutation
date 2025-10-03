import torch 
import torch.nn as nn
import random as rd
import numpy as np
from .loss_func import sce_loss
from functools import partial

class GINencoder(nn.Module):
    def __init__(self, feats_in: int = None, 
                 feats_out: int = None,
                 dropout: float = 0.5,
                 residual: bool = False):
        
        super(GINencoder, self).__init__()

        self.dropout = dropout
        self.residual = residual
        self.feats_in = feats_in
        self.feats_out = feats_out
        
        self.mlp = nn.Sequential(
            nn.Linear(feats_in, feats_out),
            nn.BatchNorm1d(feats_out),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feats_out, feats_out),
            nn.BatchNorm1d(feats_out),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
 
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

        """
        Forward pass with 3D batched inputs.
        
        Args:
            adj: Adjacency matrix [G, G] or [B, G, G]
            x: Node features [B, G, F] (batch, nodes, features)
            
        Returns:
            Transformed features [B, G, F_out]
        """
        
        B, G, F_in = x.shape  # batch size, number of nodes, feature dimension
        
        # Expand the adjacency matrix to match batch dimension if needed: [G, G] -> [B, G, G]
        if adj.dim() == 2:
            adj_expand = adj.expand(B, -1, -1)
        else:
            adj_expand = adj
        
        # Aggregate neighbor features via matrix multiplication: [B, G, G] @ [B, G, F] -> [B, G, F]
        agg = adj_expand @ x
        
        # Combine central node features with neighbors, scaled by (1 + eps)
        out = (1 + self.eps) * x + agg
        
        # Apply MLP to transform combined features
        rst = self.mlp(out.view(-1, out.shape[-1])).view(B, G, -1)
        
        if self.residual:
            if self.feats_in != self.feats_out:
                raise ValueError("For residual connection, feats_in must equal feats_out")
            rst = rst + x  # Residual connection

        return rst

class GraphMae(nn.Module):

    def __init__(self, 
                 feats_in: int = None, 
                 feats_out: int = None, 
                 mask_rate: float = 0.5, 
                 replace_rate: float = 0.1,
                 loss_fn: str = 'sce',
                 alpha: int = 2,
          ):
        super(GraphMae, self).__init__()

        self.feats_in = feats_in
        self.feats_out = feats_out
        self.mask_rate = mask_rate
        self.replace_rate = replace_rate
        self.mask_token_rate = mask_rate - replace_rate
        self.enc_mask_token = nn.Parameter(torch.zeros(1, feats_in))

        self.enc_layers, self.dec_layers = self._build_coders(feats_in, feats_out)
        
        # * setup loss function
        self.criterion = self._setup_loss_fn(loss_fn, alpha)

    def _build_coders(self, feats_in: int, feats_out: int, num_enc_layers: int = 5, num_dec_layers: int = 5, dropout: float = 0.5, residual: bool = False):
        # Build encoder layers
        enc_layers = []
        for i in range(num_enc_layers):
            in_dim = feats_in if i == 0 else feats_out
            out_dim = feats_out
            enc_layers.append(GINencoder(in_dim, out_dim, dropout=dropout, residual=residual))

        # Build decoder layers
        dec_layers = []
        for i in range(num_dec_layers):
            in_dim = feats_out
            out_dim = feats_in if i == num_dec_layers - 1 else feats_out
            dec_layers.append(GINencoder(in_dim, out_dim, dropout=dropout, residual=residual))

        return nn.ModuleList(enc_layers), nn.ModuleList(dec_layers)
    
    def _setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def random_masking(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Randomly mask node features (works with 3D batched input).
        
        Args:
            adj: Adjacency matrix [G, G]
            x: Node features [B, G, F]

        Returns:
            out_x: Tensor [B, G, F] where masked nodes are replaced either by
                   the mask token or by features copied from other nodes (noising).
            mask_idx: Tensor of shape (num_masked_nodes,) with indices of masked nodes.
        """

        B, num_nodes, F = x.shape
        device = x.device

        perm_nodes = torch.randperm(num_nodes, device=device)
        num_mask_nodes = int(self.mask_rate * num_nodes)

        mask_idx = perm_nodes[: num_mask_nodes]
        keep_idx = perm_nodes[num_mask_nodes: ]

        out_x = x.clone()

        # number of nodes to replace (noisy) among the masked nodes
        num_noise_nodes = int(self.replace_rate * num_mask_nodes) if self.replace_rate > 0 else 0

        # permute masked nodes to split between replace vs mask-token
        perm_mask = torch.randperm(num_mask_nodes, device=device)
        noise_idx = mask_idx[perm_mask[: num_noise_nodes]] if num_noise_nodes > 0 else torch.tensor([], dtype=torch.long, device=device)
        mask_token_idx = mask_idx[perm_mask[num_noise_nodes: ]] if num_mask_nodes - num_noise_nodes > 0 else torch.tensor([], dtype=torch.long, device=device)

        # choose source nodes for noisy replacement from the kept nodes when possible
        if num_noise_nodes > 0:
            if keep_idx.numel() >= num_noise_nodes:
                noisy_src = keep_idx[torch.randperm(keep_idx.numel(), device=device)[:num_noise_nodes]]
            else:
                # fallback: sample from all nodes (may include masked ones)
                noisy_src = torch.randperm(num_nodes, device=device)[:num_noise_nodes]

            # Replace across all batches
            out_x[:, noise_idx] = x[:, noisy_src]

        # assign mask token to the remaining masked nodes
        if mask_token_idx.numel() > 0:
            # expand enc_mask_token to match batch size and number of masked nodes: [1, F] -> [B, num_masked, F]
            out_x[:, mask_token_idx] = self.enc_mask_token.expand(B, mask_token_idx.numel(), -1)

        return out_x, mask_idx
    
    def encode(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.enc_layers:
            h = layer(adj, h)

        return h

    def decode(self, adj: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x_ = h
        for layer in self.dec_layers:
            x_ = layer(adj, x_)       
        return x_

    def forward(self, adj: torch.Tensor, x: torch.Tensor) -> tuple:

        # ---- attribute reconstruction ----
        loss = self.mask_attr_prediction(adj, x)
        loss_item = {"loss": loss.item()}
        return loss, loss_item

    def mask_attr_prediction(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Mask node features and predict them.
        
        Args:
            adj: Adjacency matrix [G, G]
            x: Node features [B, G, F]
            
        Returns:
            loss: Reconstruction loss
        """

        #  Mask node features and predict them.
        x_w_mask, mask_idx = self.random_masking(adj, x)

        # Encode
        h = self.encode(adj, x_w_mask)

        # Mask hidden representations of masked nodes (across all batches)
        h[:, mask_idx] = 0

        # Decode
        out_x = self.decode(adj, h)

        # Extract masked nodes: [B, num_masked, F]
        x_init = x[:, mask_idx]
        x_rec = out_x[:, mask_idx]

        assert x_init.shape == x_rec.shape, "Shape mismatch between original and reconstructed features"

        # Flatten to [B * num_masked, F] for loss computation
        x_init_flat = x_init.reshape(-1, x_init.shape[-1])
        x_rec_flat = x_rec.reshape(-1, x_rec.shape[-1])

        # Compute loss 
        loss = self.criterion(x_rec_flat, x_init_flat)
        return loss
    
    def embed(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

        """Return embeddings from the encoder (no decoding)."""
        h = self.encode(adj, x)
        return h
    

