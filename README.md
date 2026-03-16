# Character-Level Language Model from Scratch

This project implements a **character-level language model** using a **Multilayer Perceptron (MLP)** built entirely **from scratch with NumPy**, without using deep learning frameworks such as PyTorch or TensorFlow.

The model learns to predict the next character in a sequence and can generate new sentences after training.

The implementation includes a small neural network framework, a training pipeline, dataset utilities, and inference.

---

# Features

- Neural network implemented **from scratch**
- Custom **backpropagation**
- Custom implementations of:
  - Linear layer
  - Embedding layer
  - Batch Normalization
  - Tanh activation
  - Softmax
  - Multiclass Cross Entropy loss
  - Adam optimizer
- Custom **Dataset** and **DataLoader**
- Training and evaluation loops
- Sentence generation using probabilistic sampling
- Visualization of training metrics

---

# Model Architecture

The language model is defined in `mlp_lm.py` and uses the following architecture:

```
Embedding
↓
Flatten
↓
Linear
↓
BatchNorm
↓
Tanh
↓
Linear
↓
BatchNorm
↓
Tanh
↓
Linear
↓
Softmax
```

The embedding layer converts characters into dense vectors before passing them through the MLP.

---

# Project Structure

```
project/
│
├── data/
│   ├── data.py          # dataset loading and preprocessing
│   ├── dataset.py       # dataset abstraction
│   ├── dataloader.py    # batching utilities
│   └── datasets/        # raw datasets
│
├── inference/
│   └── generate.py      # sentence generation
│
├── nn/
│   ├── base.py          # core Module and Parameter classes
│   ├── linear.py        # Linear layer
│   ├── embedding.py     # Embedding + Flatten layers
│   ├── batchnorm.py     # Batch Normalization
│   ├── activations.py   # activation functions
│   ├── loss.py          # cross entropy loss
│   ├── optim.py         # Adam optimizer
│   └── sequential.py    # Sequential container
│
├── utils/
│   ├── plot.py          # training visualization
│   ├── save_load.py     # saving and loading model parameters
│   └── split.py         # train/test split
│
├── mlp_lm.py            # language model definition
├── train.py             # training loop
├── main.py              # training entry point
└── models/              # saved parameters
```

---

# Procedure

The main procedures defined in `main.py`:

Training Procedure:

Steps:

1. Load dataset
2. Build character vocabulary
3. Encode characters to integer indices
4. Create context-target training samples
5. Split dataset into train and test sets
6. Train the MLP model
7. Plot training metrics
8. Save model parameters

Training uses the following components:

- Loss: Multiclass Cross Entropy  
- Optimizer: Adam

---

Inference Procedure:

1. Load dataset
2. Build character vocabulary
3. Encode characters to integer indices
4. Load model parameters
5. Generate sentences


# Running the Project

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run inference

```bash
python main.py
```

If you want to try training the model you should call the function train_model()inside of main()

The script will:

1. Load model
2. Generate example sentences

---

# Example Generation

The model generates sentences by sampling from the predicted probability distribution of the next character using a multinomial sampler.

---

# Learning Goals

The project was done to gain a deeper understanding of how deep learning systems work internally by implementing:

- forward passes
- backpropagation
- parameter updates
- training loops
- dataset pipelines

All components are written **without high-level ML frameworks**. 

Just Python and Numpy.
