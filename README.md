# Neural Network from Scratch

This project implements a simple neural network from scratch using NumPy. The network is trained on a spiral dataset to classify points into three classes. It includes fundamental components such as dense layers, activation functions, an optimizer (Adam), and a loss function (categorical cross-entropy). 

⬇️ Check out how I used what I learned in this project to build something more complex and meaningful here 
https://github.com/bryson32/DeepLearning-LipReader 

## Features
- Fully connected (dense) layers
- ReLU and Softmax activation functions
- Adam optimizer for weight updates
- Categorical cross-entropy loss
- Backpropagation for gradient calculation
- Simple training loop with accuracy tracking

## Requirements
Ensure you have the following dependencies installed:

```bash
pip install numpy matplotlib nnfs
```

## Usage
Run the script to train the neural network:

```bash
python core.py
```

The model trains for 10,000 epochs and prints the loss and accuracy every 1,000 epochs.

## Structure
- `Layer_Dense`: Implements a fully connected layer.
- `Activation_ReLU`: Implements the ReLU activation function.
- `Activation_Softmax`: Implements the Softmax function.
- `Loss_crossentropy`: Computes categorical cross-entropy loss.
- `Optimizer_Adam`: Implements the Adam optimization algorithm.

## Training Process
1. Generate spiral data using `nnfs.datasets.spiral_data()`.
2. Pass data through two hidden layers with ReLU activations.
3. Use Softmax for the output layer to produce class probabilities.
4. Compute loss using categorical cross-entropy.
5. Perform backpropagation to update weights using Adam optimizer.
