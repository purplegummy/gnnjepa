# GNN-JEPA

A Graph Neural Network implementation of JEPA (Joint Embedding Predictive Architecture) for epidemic modeling on SIR graphs.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Generate data:
```bash
python util/generate_data.py
```

Train:
```bash
python train.py
```

## Architecture

- **Encoder** (`model/encoder.py`): 2-layer GCN that maps SIR node states to latent embeddings
- **Predictor** (`model/predictor.py`): 2-layer GCN that takes current embedding + vaccination action and predicts next embedding
- **JEPA** (`model/jepa.py`): combines encoder and predictor; encoder provides state representations, predictor handles action-conditional reasoning
- **SIGReg** (`model/sigreg.py`): Sketch Isotropic Gaussian Regularizer to prevent representational collapse

## Data

A single SIR epidemic time series on a fixed Erdos-Renyi contact graph. Each transition is `(graph_t, action, graph_t+1)` where action is a per-node binary vaccination mask.

Epoch 50/50  train_mse=0.4608  val_mse=0.4619  R²: cases=0.430  deaths=0.410  stringency=0.731
Epoch 50/50  train_mse=0.5601  val_mse=0.5632  R²: cases=-0.083  deaths=0.582  stringency=0.756


LINEAR ONLY: Epoch 50/50  train_mse=0.8359  val_mse=0.9939  R²: cases=-0.516  deaths=0.677  stringency=0.081
ENCODER+LINEAR: Epoch 50/50  train_mse=0.5104  val_mse=0.5148  R²: cases=0.505  deaths=0.359  stringency=0.673