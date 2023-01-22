import numpy as np


class NeuralNetwork():
    def __init__(self):
        self.weights = np.random.random((3, 1))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    def _extract_result(self, output: list):
        return [1 if x > 0.5 else 0 for x in output]

    def train(self, training_inputs, training_outputs, training_iterations):
        for i in range(training_iterations):
            input_layer = training_inputs
            outputs = self._sigmoid(np.dot(input_layer, self.weights))
            error = training_outputs - outputs
            adjustments = error * self._sigmoid_derivative(outputs)
            self.weights += np.dot(input_layer.T, adjustments)

    def decide(self, inputs):
        return self._extract_result(self._sigmoid(np.dot(inputs, self.weights)))


if __name__ == '__main__':
    neural_network = NeuralNetwork()
    print('The initial weights are:\n', neural_network.weights, "\n")

    training_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])

    training_outputs = np.array([[0, 1, 1, 0]]).T

    neural_network.train(training_inputs, training_outputs, 90000)
    print('Weights after training: \n', neural_network.weights)

    tests = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 0]])
    print("Test Inputs:\n", tests, "\n")
    print("Test Outputs:\n", neural_network.decide(tests), "\n")
