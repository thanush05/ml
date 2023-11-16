import numpy as np

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - x**2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return 1.0 * (x > 0)

# Define the neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation_function):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases with random values
        self.weights_input_hidden = np.random.uniform(-1, 1, (self.input_size, self.hidden_size))
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.uniform(-1, 1, (self.hidden_size, self.output_size))
        self.bias_output = np.zeros((1, self.output_size))

        # Choose activation function
        self.activation = activation_function
        if activation_function == 'sigmoid':
            self.activation_fn = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation_function == 'tanh':
            self.activation_fn = tanh
            self.activation_derivative = tanh_derivative
        elif activation_function == 'relu':
            self.activation_fn = relu
            self.activation_derivative = relu_derivative

    def forward(self, X):
        # Input to hidden layer
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.activation_fn(self.hidden_input)

        # Hidden layer to output
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = self.activation_fn(self.output_input)

        return self.output

    def backward(self, X, y, learning_rate):
        # Calculate error
        self.error = y - self.output

        # Backpropagate the error
        delta_output = self.error * self.activation_derivative(self.output)
        delta_hidden = delta_output.dot(self.weights_hidden_output.T) * self.activation_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(delta_output) * learning_rate
        self.bias_output += np.sum(delta_output, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += X.T.dot(delta_hidden) * learning_rate
        self.bias_hidden += np.sum(delta_hidden, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)

            # Backward pass and weight updates
            self.backward(X, y, learning_rate)

            # Calculate and print loss
            loss = np.mean(np.square(y - output))
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}: Loss = {loss}")

if __name__ == "__main__":
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Experiment with different numbers of nodes in the hidden layer and activation functions
    hidden_layer_sizes = [2, 4, 8]  # You can change these values
    activation_functions = ['sigmoid', 'tanh', 'relu']  # You can change these values

    for hidden_size in hidden_layer_sizes:
        for activation_function in activation_functions:
            print(f"Hidden Layer Size: {hidden_size}, Activation Function: {activation_function}")
            model = NeuralNetwork(input_size=2, hidden_size=hidden_size, output_size=1, activation_function=activation_function)
            model.train(X, y, epochs=10000, learning_rate=0.1)

            # Test the model
            test_input = X
            predicted_output = model.forward(test_input)
            print("Predicted Output:")
            print(predicted_output)