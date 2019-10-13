
from typing import List
import numpy as np
from bbnet.layer import Layer
from bbnet.tensor import Tensor


class NeuralNetwork:
    """
    It contains a list of Layers and make things with them so it can become an amazing AI
    """

    def __init__(self, layers: List[Layer], learning_rate=0.1):
        self.layers: List[Layer] = layers
        self.learning_rate: float = learning_rate

    def add_layer(self, layer: Layer) -> None:
        self.layers.append(layer)

    def set_layers(self, layers: List[Layer]):
        self.layers = layers

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def guess(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)

    def backward(self, inputs: Tensor, target: Tensor) -> Tensor:
        guessed = self.forward(inputs)
        for index in range(len(self.layers)-1, -1, -1):
            layer = self.layers[index]
            if index == len(self.layers) - 1:
                layer.error = target - guessed
                layer.delta = layer.error * layer.deriv_func(guessed)
            else:
                next_layer = self.layers[index+1]
                layer.error = next_layer.weights @ next_layer.delta
                layer.delta = layer.error * layer.deriv_func(layer.inputs)

        for index in range(len(self.layers)):
            layer = self.layers[index]
            # if it's the first layer, we should use the original inputs, otherwise we take the inputs from the feedforward method
            input_to_use: Tensor = np.atleast_2d(
                inputs if index == 0 else self.layers[index-1].inputs)
            layer.weights += layer.delta * input_to_use.T * self.learning_rate
        return guessed

    def train(self, inputs: Tensor, targets: Tensor, epochs=10000, verbose=True, log_each=500) -> None:
        for epoch in range(epochs):
            for inpt, target in zip(inputs, targets):
                self.backward(inpt, target)
            if epoch % log_each == 0 and verbose:
                print(f'Epoch: {epoch}')
