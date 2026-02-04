# M4 Assignment 1: Understanding Neural Networks Through SGD and Attention

This project consists of two parts that provide hands-on understanding of fundamental neural network concepts:
- **Part A**: Manual implementation and validation of Stochastic Gradient Descent (SGD)
- **Part B**: Implementation of self-attention mechanisms to demonstrate context-dependent word representations

## Table of Contents
- [Overview](#overview)
- [Part A: Manual SGD](#part-a-manual-sgd)
- [Part B: Attention Contextualization](#part-b-attention-contextualization)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Requirements](#requirements)

## Overview

This assignment explores the inner workings of neural network training by manually computing gradients and implementing attention mechanisms. The goal is to demystify the backpropagation process and understand how context affects word embeddings through self-attention.

## Part A: Manual SGD

### Description

Part A demystifies the training process by manually computing the forward pass, loss, and gradients to understand how backpropagation drives parameter updates. This implementation uses a simple linear neural network with one weight parameter trained on the Swedish Auto Insurance dataset.

### Dataset & Hyperparameters

- **Dataset**: Swedish Auto Insurance Dataset (first 3 samples used for manual computation)
- **Model**: Simple linear model: `ŷ = x · w`
- **Loss Function**: Mean Squared Error (MSE): `L = (t - ŷ)²`
- **Hyperparameters**:
  - Learning Rate (α): 2.0 (default, configurable)
  - Initial Weight (w): 10.0 (default, configurable)

### Mathematical Steps

The training process follows these steps for each sample:

1. **Forward Pass**: `ŷ = x · w`
2. **Loss Calculation**: `L = (t - ŷ)²`
3. **Gradient Computation**: `∂L/∂w = 2x(ŷ - t)`
4. **Weight Update**: `w_new = w_old - α · (∂L/∂w)`

### Manual Computation

The file `manualCompute.ods` contains a spreadsheet with manually computed values for the first 3 samples, showing:
- Initial weight (w_old)
- Input value (x)
- Predicted output (ŷ)
- Gradient (dL/dw)
- Updated weight (w_new)

This manual computation is verified against the code implementation to ensure correctness.

### Features

- **Batch Gradient Descent**: Updates weights using the entire dataset
- **Stochastic Gradient Descent**: Updates weights after each sample
- **Interactive Dashboard**: Visualizes training progress with loss curves, weight updates, and model fit
- **Configurable Parameters**: Learning rate, initial weight, number of epochs, and training mode

## Part B: Attention Contextualization

### Description

Part B demonstrates how self-attention mechanisms allow models to capture context-dependent meanings of words. Static embeddings fail to differentiate meanings of homonyms (words with multiple meanings), while attention-based representations adapt based on context.

### Task

The implementation uses the homonym "fine" in two different contexts:
1. "This sample aged like fine wine" (meaning: high quality)
2. "I just got a parking fine" (meaning: penalty)

### Implementation Details

1. **Embeddings**: Initialize random 2D vectors for each unique token using PyTorch
2. **Attention Mechanism**:
   - Query (Q), Key (K), Value (V) matrices are all equal to embeddings (E)
   - Attention scores: `A = softmax(Q · K^T)`
   - Attention output: `output = A · V`
3. **Validation**: Uses cosine similarity to show how the representation of "fine" differs between contexts

### Mathematical Formula

The self-attention mechanism computes:
```
Attention(Q, K, V) = softmax(QK^T) · V
```

Where:
- Q = K = V = E (embeddings)
- Softmax normalizes attention scores across all tokens
- The output is a context-aware representation

### Visualization

The implementation creates scatter plots showing:
- Attention output embeddings for both sentences
- How the same word ("fine") has different representations in different contexts
- Cosine similarity metrics before and after attention

## Installation

### Prerequisites
- Python 3.12 or higher
- pip or uv package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Dun3AAU/M4_Assignment_1.git
cd M4_Assignment_1
```

2. Install dependencies:

Using pip:
```bash
pip install -r requirements.txt
```

Or using uv (recommended):
```bash
uv sync
```

## Usage

### Part A: Manual SGD

#### Command Line Interface

Run batch gradient descent:
```bash
python PartA.py --mode batch --epochs 5 --learning_rate 2.0 --weight 10
```

Run stochastic gradient descent:
```bash
python PartA.py --mode stochastic --epochs 5 --learning_rate 2.0 --weight 10
```

#### Arguments
- `--mode`: Training mode (`batch` or `stochastic`, default: `batch`)
- `--epochs`: Number of training epochs (default: `5`)
- `--learning_rate`: Learning rate for weight updates (default: `2.0`)
- `--weight`: Initial weight value (default: `10`)

#### Interactive Dashboard

Launch the interactive Dash application:
```bash
python app.py
```

Then open your browser to `http://127.0.0.1:8050/` to access the dashboard.

The dashboard provides:
- Real-time visualization of training progress
- Loss vs. epoch plots
- Weight evolution over epochs
- Dataset and model fit visualization
- Loss landscape (loss as a function of weight)
- Configurable hyperparameters through UI controls
- Training history table

### Part B: Attention Contextualization

Run the attention visualization:
```bash
python PartB.py
```

This will:
1. Initialize embeddings for all unique words in both sentences
2. Compute self-attention for each sentence
3. Display scatter plots showing attention output embeddings
4. Print cosine similarity metrics to demonstrate context-dependent representations
5. Open interactive Plotly visualizations in your browser

## File Structure

```
M4_Assignment_1/
│
├── README.md                 # This file - project documentation
├── pyproject.toml           # Project dependencies and metadata
├── uv.lock                  # Locked dependencies (uv package manager)
│
├── manualCompute.ods        # Spreadsheet with manual SGD calculations (Part A)
│
├── PartA.py                 # Command-line SGD implementation
│                           # Supports batch and stochastic gradient descent
│
├── app.py                   # Interactive Dash dashboard for Part A
│                           # Visualizes training process with interactive controls
│
└── PartB.py                 # Attention mechanism implementation
                            # Demonstrates context-dependent embeddings
```

## Requirements

### Core Dependencies
- **numpy**: Numerical computing
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning utilities (MinMaxScaler, cosine similarity)
- **torch**: PyTorch for embeddings generation
- **plotly**: Interactive visualizations
- **dash**: Web-based dashboard framework
- **seaborn**: Statistical data visualization
- **matplotlib**: Static plotting

## Key Concepts Demonstrated

### Part A - Neural Network Training
- Forward propagation
- Loss computation (Mean Squared Error)
- Gradient calculation using calculus
- Backpropagation and weight updates
- Batch vs. Stochastic gradient descent
- Learning rate impact on convergence
- Loss landscape visualization

### Part B - Attention Mechanisms
- Word embeddings
- Self-attention computation
- Softmax normalization
- Context-dependent representations
- Homonym disambiguation
- Cosine similarity for semantic comparison

## Educational Value

This project provides hands-on experience with:
1. **Manual Computation**: Understanding gradient descent at a fundamental level
2. **Verification**: Comparing manual calculations with code implementation
3. **Visualization**: Seeing how parameters evolve during training
4. **Attention Mechanism**: Understanding how modern NLP models capture context
5. **Interactive Learning**: Experimenting with hyperparameters through the dashboard

## References

- Swedish Auto Insurance Dataset: [Raw Data File (CSV)](https://raw.githubusercontent.com/aaubs/ds-master/main/data/Swedish_Auto_Insurance_dataset.csv)
- Attention Mechanism: "Attention Is All You Need" (Vaswani et al., 2017)
- Gradient Descent: Standard optimization technique in machine learning

## License

This project is created for educational purposes as part of M4 Assignment 1.
