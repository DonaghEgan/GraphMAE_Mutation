# GraphMAE Mutation Analysis

This project implements a Graph Masked Autoencoder (GraphMAE) for analyzing mutation data from cancer genomics studies.

## Project Structure

- `data/`: Contains raw and processed data
- `notebooks/`: Jupyter notebooks for exploration and analysis
- `scripts/`: Additional utility scripts
- `src/`: Source code for the project
  - `data_loader.py`: Data loading and preprocessing
  - `model.py`: GraphMAE model definition
  - `trainer.py`: Training logic
  - `evaluator.py`: Evaluation metrics and utilities
  - `loss_func.py`: Custom loss functions
  - `download_study.py`: Study data download utilities
  - `process_data.py`: Data processing utilities
- `tests/`: Unit tests
- `config.py`: Centralized configuration management
- `main.py`: Main training script
- `requirements.txt`: Project dependencies

## Installation

1. Clone the repository:
```bash
git clone https://github.com/DonaghEgan/GraphMAE_Mutation.git
cd GraphMAE_Mutation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

Run the training pipeline with default settings from `config.py`:

```bash
python main.py
```

### Custom Configuration

Override default settings via command-line arguments:

```bash
python main.py --epochs 300 --batch-size 64 --learning-rate 0.001
```

### Advanced Options

```bash
# Train with custom study data
python main.py --study msk_pan_2017

# Save node embeddings after training
python main.py --save-embeddings

# Full example with multiple options
python main.py --epochs 500 --batch-size 32 --learning-rate 0.0005 --save-embeddings
```

### Command-Line Arguments

- `--epochs`: Number of training epochs (default: from config)
- `--batch-size`: Batch size for training (default: from config)
- `--learning-rate`: Learning rate for optimizer (default: from config)
- `--study`: Study name to download and process (default: from config)
- `--save-embeddings`: Save node embeddings after training

## Configuration

Edit `config.py` to modify default settings for:

- **Model Architecture**: Feature dimensions, dropout, mask rates
- **Training**: Epochs, learning rate, batch size, early stopping
- **Data**: Study selection, train/val/test splits, random seed
- **Paths**: Output directories for results and models

## Output

Training produces the following outputs in the `results/` directory:

- `logs/`: Training logs with timestamps
- `configs/`: Saved configurations for each run
- `results/`: Training metrics and history
- `models/`: Model checkpoints (best_model.pt)
- `embeddings/`: Node embeddings (if --save-embeddings used)

## Testing

Run unit tests:

```bash
pytest tests/
```

## Project Details

This project uses a Graph Masked Autoencoder approach for learning representations from cancer mutation data. The model:

- Masks node features randomly during training
- Reconstructs masked features using graph structure
- Learns robust node embeddings for downstream tasks
- Supports multiple loss functions (MSE, SCE)
