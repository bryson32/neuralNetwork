import sys
import numpy as np
import matplotlib
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# X = [[1, 2, 3, 2.5],    #batch of 3, X is formal variable for inputs
#      [2.0, 5.0, -1.0, 2.0],
#      [-1.5, 2.7, 3.3, -0.8]]

# X, Y = spiral_data(100, 3) #100 points per class, 3 classes

# # Combined Softmax Activation + Cross Entropy Loss
# class Activation_Softmax_Loss_CategoricalCrossentropy():
#     def backward(self, dvalues, y_true):
#         # Number of samples
#         samples = len(dvalues)
        
#         # If labels are one-hot encoded, turn them into discrete values
#         if len(y_true.shape) == 2:
#             y_true = np.argmax(y_true, axis=1)
        
#         # Copy so we can safely modify
#         self.dinputs = dvalues.copy()
#         # Calculate gradient
#         self.dinputs[range(samples), y_true] -= 1
#         # Normalize gradient
#         self.dinputs = self.dinputs / samples


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dvalues):
        # Gradients with respect to weights
        self.dweights = np.dot(self.inputs.T, dvalues)
        # Gradients with respect to biases
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradients with respect to inputs (for the layer below)
        self.dinputs = np.dot(dvalues, self.weights.T)
    
    
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0    

class Activation_Softmax:    #goes on outermot layer to provide the probabilities
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims=True)
        self.output = probabilities
    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.zeros_like(dvalues)
        # For each sample, compute the Jacobian matrix
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output
            single_output = single_output.reshape(-1, 1)
            # Jacobian matrix of the softmax
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient and store
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
        
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_crossentropy(Loss):
    def forward(self, y_predicted, y_true):
        samples = len(y_predicted)
        y_pred_clipped = np.clip(y_predicted, 1e-7, 1-1e-7)
        
        if len(y_true.shape) == 1:  #one hot encoded vectors
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:  #scalar cclass values
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        
        neg_log_likelihoods = -np.log(correct_confidences)
        return neg_log_likelihoods
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize
        self.dinputs = self.dinputs / samples
        
# Generate data
X, Y = spiral_data(100, 3)

# Create layers
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

# Loss function
loss_function = Loss_crossentropy()

# Training params
learning_rate = 0.1
epochs = 10001

for epoch in range(epochs):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Compute loss
    loss = loss_function.calculate(activation2.output, Y)

    # (Optional) measure accuracy
    predictions = np.argmax(activation2.output, axis=1)
    if len(Y.shape) == 2:
        Y_labels = np.argmax(Y, axis=1)
    else:
        Y_labels = Y
    accuracy = np.mean(predictions == Y_labels)

    # Print every 1000 epochs
    if not epoch % 1000:
        print(f"epoch: {epoch}, loss: {loss:.3f}, accuracy: {accuracy:.3f}")

    # Backward pass
    # 1) Loss backward
    loss_function.backward(activation2.output, Y)
    # 2) Softmax backward
    activation2.backward(loss_function.dinputs)
    # 3) Dense2 backward
    dense2.backward(activation2.dinputs)
    # 4) ReLU backward
    activation1.backward(dense2.dinputs)
    # 5) Dense1 backward
    dense1.backward(activation1.dinputs)

    # Update weights and biases
    dense1.weights -= learning_rate * dense1.dweights
    dense1.biases  -= learning_rate * dense1.dbiases
    dense2.weights -= learning_rate * dense2.dweights
    dense2.biases  -= learning_rate * dense2.dbiases