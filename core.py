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

class Optimizer_Adam:
    """
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7
    """
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.iterations = 0

    def update_params(self, layer):
        # If layer does not have momentums/ cache arrays, create them
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum with current gradients
        layer.weight_momentums = (
            self.beta_1 * layer.weight_momentums
            + (1 - self.beta_1) * layer.dweights
        )
        layer.bias_momentums = (
            self.beta_1 * layer.bias_momentums
            + (1 - self.beta_1) * layer.dbiases
        )

        # Corrected momentums
        weight_momentums_corrected = (
            layer.weight_momentums
            / (1 - self.beta_1 ** (self.iterations + 1))
        )
        bias_momentums_corrected = (
            layer.bias_momentums
            / (1 - self.beta_1 ** (self.iterations + 1))
        )

        # Update cache with squared gradients
        layer.weight_cache = (
            self.beta_2 * layer.weight_cache
            + (1 - self.beta_2) * (layer.dweights ** 2)
        )
        layer.bias_cache = (
            self.beta_2 * layer.bias_cache
            + (1 - self.beta_2) * (layer.dbiases ** 2)
        )

        # Corrected cache
        weight_cache_corrected = (
            layer.weight_cache
            / (1 - self.beta_2 ** (self.iterations + 1))
        )
        bias_cache_corrected = (
            layer.bias_cache
            / (1 - self.beta_2 ** (self.iterations + 1))
        )

        # Vanilla Adam parameter update
        layer.weights -= self.learning_rate * weight_momentums_corrected / (
            np.sqrt(weight_cache_corrected) + self.epsilon
        )
        layer.biases -= self.learning_rate * bias_momentums_corrected / (
            np.sqrt(bias_cache_corrected) + self.epsilon
        )

    def post_update_params(self):
        # Increment iteration after updating
        self.iterations += 1

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
            
        dvalues_clipped = np.clip(dvalues, 1e-7, 1 - 1e-7)
        
        # Calculate gradient
        self.dinputs = -y_true / dvalues_clipped
        # Normalize
        self.dinputs = self.dinputs / samples
        
# Generate data
X, Y = spiral_data(100, 3)

# Create layers
dense1 = Layer_Dense(2, 64)    # 64 neurons
activation1 = Activation_ReLU()

dense2 = Layer_Dense(64, 64)   # second hidden layer
activation2 = Activation_ReLU()

dense3 = Layer_Dense(64, 3)    # output layer
activation3 = Activation_Softmax()

# Loss function
loss_function = Loss_crossentropy()
optimizer = Optimizer_Adam(learning_rate=0.01)

# Training params
epochs = 10001

for epoch in range(epochs):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)

    # Compute loss
    loss = loss_function.calculate(activation3.output, Y)

    # (Optional) measure accuracy
    predictions = np.argmax(activation3.output, axis=1)
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
    loss_function.backward(activation3.output, Y)
    # 2) Softmax backward
    activation3.backward(loss_function.dinputs)
    # 3) Dense2 backward
    dense3.backward(activation3.dinputs)
    # 4) ReLU backward
    activation2.backward(dense3.dinputs)
    # 5) Dense1 backward
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)

    dense1.backward(activation1.dinputs)
    
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()