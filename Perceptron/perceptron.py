# Let's create a perceptron to recognize the function f(a,b,c) = a, where a, b, c belongs to {0,1}
# only one neuron
import numpy as np


# import random


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def extract_result(output: list):
    return [1 if x > 0.5 else 0 for x in output]


training_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])

training_outputs = np.array([[0, 1, 1, 0]]).T

# if we want to define a seed in the future:
# np.random.seed(1)

# defining the bounds for the limits as [0,1].
weights = np.random.random((3, 1))

print("The initial weights are: \n", weights, "\n")

# calculate the erros and adjust the weights accordingly. Bigger errors leads to a bigger adjust.
# Besides, we are going to use the derivative of the sigmoid function to help to adjust the weights. That's because
# the derivative of this function in big numbers of the x axis is smaller than in small numbers (considering the
# absolute value of this number), indicating a higher confidence. Basically, we will use the following function to
# adjust our weights:
#                               error * input * derivative(sigmoid(output))
for i in range(90000):
    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, weights))
    error = training_outputs - outputs
    adjustments = error * sigmoid_derivative(outputs)
    weights += np.dot(input_layer.T, adjustments)

print("Weights after training: \n", weights, "\n")

# After learning let's try it:

tests = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 0]])

outputs = extract_result(sigmoid(np.dot(tests, weights)))
print("Test Inputs:\n", tests, "\n")
print("Test Outputs:\n", outputs, "\n")
