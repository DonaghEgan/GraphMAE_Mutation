import torch 
import torch.nn as nn
import random as rd
import numpy as np

class GINencoder(nn.Module):
    def __init__(self, feats_in: int = None, feats_out: int = None, dropout: float = 0.5, residual: bool = False):
        super(GINencoder, self).__init__()

        self.feats_in = feats_in
        self.feats_out = feats_out
        self.dropout = dropout
        self.residual = residual

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

        B, G, F_in = x.shape # batch size, number of nodes, feature dimension

        # Expand the adjacency matrix to match batch dimension: [G, G] -> [B, G, G]
        adj_expand = adj.expand(B, -1, -1)

        # Aggregate neighbor features via matrix multiplication: [B, G, G] @ [B, G, F] -> [B, G, F]
        agg = adj_expand @ x

        # Combine central node features with neighbors, scaled by (1 + eps)
        out = (1 + self.eps) * x + agg
        
        # Apply MLP to transform combined features
        rst = self.mlp(out.view(-1, out.shape[-1])).view(x.shape[0], x.shape[1], -1)

        if self.residual:
            if self.feats_in != self.feats_out:
                raise ValueError("For residual connection, feats_in must equal feats_out")
            rst = rst + x  # Residual connection

        return rst

class GraphMae(nn.Module):
    def __init__(self, feats_in: int = None, feats_out: int = None, mask_rate: float = 0.5, replace_rate: float = 0.1):
        super(GraphMae, self).__init__()

        self.feats_in = feats_in
        self.feats_out = feats_out
        self.mask_rate = mask_rate
        self.replace_rate = replace_rate
        self.mask_token_rate = mask_rate - replace_rate
        self.enc_mask_token = nn.Parameter(torch.zeros(1, feats_in))

    def random_masking(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        
        """Randomly mask node features.

        Returns:
            out_x: Tensor with same shape as x where masked nodes are replaced either by
                   the mask token or by features copied from other nodes (noising).
            mask: Bool tensor of shape (num_nodes,) where True indicates a masked node.
            replace_idx: Bool tensor of shape (num_nodes,) where True indicates a replaced (noisy) node.
        """

        num_nodes = adj.shape[0]
        device = x.device

        perm_nodes = torch.randperm(num_nodes, device=device)
        num_mask_nodes = int(self.mask_rate * num_nodes)

        if num_mask_nodes == 0:
            # nothing to mask
            mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            replace_idx = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            return x.clone(), mask, replace_idx

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

            out_x[noise_idx] = x[noisy_src]

        # assign mask token to the remaining masked nodes
        if mask_token_idx.numel() > 0:

            # expand enc_mask_token to match number of masked rows
            out_x[mask_token_idx] = self.enc_mask_token.expand(mask_token_idx.numel(), -1)

        # boolean masks for callers
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        mask[mask_idx] = True
        replace_idx = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        if num_noise_nodes > 0:
            replace_idx[noise_idx] = True

        return out_x, mask, replace_idx



