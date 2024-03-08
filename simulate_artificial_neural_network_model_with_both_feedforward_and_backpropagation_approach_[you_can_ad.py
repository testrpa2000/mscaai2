from math import exp
from random import seed, random

def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for _ in range(n_inputs + 1)]} for _ in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for _ in range(n_hidden + 1)]} for _ in range(n_outputs)]
    network.append(output_layer)
    return network

def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation

def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

def transfer_derivative(output):
    return output * (1.0 - output)

def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

def update_weights(network, row, learn_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += learn_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += learn_rate * neuron['delta']

def train_network(network, train, learn_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for _ in range(n_outputs)]
            expected[int(row[-1])] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, learn_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learn_rate, sum_error))

def back_propagation(train, test, l_rate, n_epoch, n_hidden):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, l_rate, n_epoch, n_outputs)
    predictions = [predict(network, row) for row in test]
    return predictions

# User Input
seed(1)
hidden_neurons = int(input("Enter the number of hidden neurons: "))
learning_rate = float(input("Enter the learning rate: "))
epochs = int(input("Enter the number of training epochs: "))

# Example Dataset
dataset = [
    [2.7810836, 2.550537003, 0],
    [1.465489372, 2.362125076, 0],
    [3.396561688, 4.400293529, 0],
    [1.38807019, 1.850220317, 0],
    [3.06407232, 3.005305973, 0],
    [7.627531214, 2.759262235, 1],
    [5.332441248, 2.088626775, 1],
    [6.922596716, 1.77106367, 1],
    [8.675418651, -0.242068655, 1],
    [7.673756466, 3.508563011, 1]
]

n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = initialize_network(n_inputs, hidden_neurons, n_outputs)

# Feedforward
print("Feedforward:")
for row in dataset:
    output = forward_propagate(network, row)
    print(f"Input: {row[:-1]}, Predicted Output: {output}")

# Backpropagation
print("\nBackpropagation:")
for epoch in range(epochs):
    sum_error = 0
    for row in dataset:
        outputs = forward_propagate(network, row)
        expected = [0 for _ in range(n_outputs)]
        expected[int(row[-1])] = 1
        sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
        backward_propagate_error(network, expected)
        update_weights(network, row, learning_rate)
    print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learning_rate, sum_error))

# Print final weights
print("\nFinal Weights:")
for i, layer in enumerate(network):
    print(f"\nLayer {i + 1} Weights:")
    for j, neuron in enumerate(layer):
        print(f"Neuron {j + 1} Weights: {neuron['weights']}")




# 1. **Import Libraries**: The code starts by importing necessary libraries. `math.exp` is imported to calculate the exponential function, and `random.seed` and `random.random` are imported to generate random numbers.

# 2. **Initialize Network**: The `initialize_network` function is defined to create a neural network with random weights. It takes three arguments: `n_inputs` (number of input neurons), `n_hidden` (number of neurons in the hidden layer), and `n_outputs` (number of output neurons). It initializes the network with random weights for each neuron in the hidden and output layers.

# 3. **Activation and Transfer Functions**: The `activate` function calculates the activation of a neuron by taking the dot product of its weights and inputs and adding the bias. The `transfer` function applies the sigmoid activation function to the activation to produce the neuron's output.

# 4. **Forward Propagation**: The `forward_propagate` function calculates the output of each neuron in the network by propagating the inputs forward through the network layer by layer. It iterates over each layer and calculates the output of each neuron using the activation and transfer functions.

# 5. **Prediction**: The `predict` function predicts the class of a given input row by forward propagating the inputs through the network and returning the index of the neuron with the highest output in the output layer.

# 6. **Transfer Derivative Function**: The `transfer_derivative` function calculates the derivative of the sigmoid activation function with respect to the neuron's output.

# 7. **Backward Propagation**: The `backward_propagate_error` function calculates the error for each neuron in the network and propagates it backward through the network to update the weights. It iterates over each layer in reverse order and calculates the error for each neuron based on the errors from the next layer.

# 8. **Update Weights**: The `update_weights` function updates the weights of each neuron in the network based on the error and the input values. It iterates over each layer and each neuron within the layer and updates the weights using the error, learning rate, and input values.

# 9. **Train Network**: The `train_network` function trains the neural network on a given dataset for a specified number of epochs. It iterates over each epoch and each row in the dataset, calculates the error, performs backward propagation, and updates the weights.

# 10. **Back Propagation Algorithm**: The `back_propagation` function is the main entry point for training the neural network using the backpropagation algorithm. It initializes the network, trains it on the training dataset, and returns the predictions for the test dataset.

# 11. **User Input and Example Dataset**: The user is prompted to input the number of hidden neurons, learning rate, and number of training epochs. An example dataset is provided for demonstration purposes.

# 12. **Feedforward and Backpropagation Output**: The code prints the results of feedforward propagation for each input in the dataset, followed by the progress of backpropagation during training. It displays the epoch number, learning rate, and total error for each epoch.

# 13. **Print Final Weights**: Finally, the code prints the final weights of each neuron in each layer after training.

# This code implements a basic neural network using the backpropagation algorithm for a classification task. It demonstrates how to initialize the network, train it on a dataset, and make predictions.