"""
Example demonstrating the two modes of read_reactome_new function.

This shows how to create adjacency matrices either for:
1. ALL genes found in Reactome (useful for full network analysis)
2. Only genes in your specific gene_index (useful for filtered datasets)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.read_specific import read_reactome_new

# Example gene index (in practice, this would come from your data)
example_gene_index = {
    'TP53': 0,
    'BRCA1': 1,
    'EGFR': 2,
    'KRAS': 3,
    'PIK3CA': 4,
    # ... more genes
}

print("=" * 80)
print("MODE 1: All Genes - Create adjacency matrix for ALL genes in Reactome")
print("=" * 80)

# This returns BOTH the adjacency matrix AND a new gene_index dictionary
adj_matrix_all, all_genes_index = read_reactome_new(
    gene_index=example_gene_index,  # Not actually used in this mode
    folder='data/',
    mode='all_genes'
)

print(f"\nResults:")
print(f"  - Adjacency matrix shape: {adj_matrix_all.shape}")
print(f"  - Number of genes: {len(all_genes_index)}")
print(f"  - Number of edges: {adj_matrix_all.sum().item() / 2:.0f}")  # Divide by 2 since symmetric
print(f"  - First 5 genes: {list(all_genes_index.keys())[:5]}")
print(f"  - Density: {adj_matrix_all.sum().item() / (adj_matrix_all.shape[0] ** 2):.4f}")

print("\n" + "=" * 80)
print("MODE 2: Filtered Genes - Create adjacency matrix ONLY for genes in gene_index")
print("=" * 80)

# This returns ONLY the adjacency matrix, filtered to your gene_index
adj_matrix_filtered = read_reactome_new(
    gene_index=example_gene_index,
    folder='data/',
    mode='filtered_genes'
)

print(f"\nResults:")
print(f"  - Adjacency matrix shape: {adj_matrix_filtered.shape}")
print(f"  - Number of genes: {len(example_gene_index)}")
print(f"  - Number of edges: {adj_matrix_filtered.sum().item() / 2:.0f}")
print(f"  - Genes included: {list(example_gene_index.keys())}")
print(f"  - Density: {adj_matrix_filtered.sum().item() / (adj_matrix_filtered.shape[0] ** 2):.4f}")

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"\nAll genes mode:      {adj_matrix_all.shape[0]} genes")
print(f"Filtered genes mode: {adj_matrix_filtered.shape[0]} genes")
print(f"Difference:          {adj_matrix_all.shape[0] - adj_matrix_filtered.shape[0]} genes excluded")

print("\n" + "=" * 80)
print("WHEN TO USE EACH MODE")
print("=" * 80)
print("""
Use 'all_genes' mode when:
  - You want to analyze the complete Reactome network
  - You're doing exploratory analysis
  - You want to include all genes with high-confidence interactions (score >= 0.9)
  - You'll filter genes later based on other criteria

Use 'filtered_genes' mode when:
  - You have a specific set of genes from your mutation data
  - You want to match the adjacency matrix to your feature matrix dimensions
  - You're training a GNN and need aligned gene sets
  - You want to focus only on genes present in your dataset
""")
