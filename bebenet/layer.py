from typing import Callable, Any
from bbnet.tensor import Tensor
import numpy as np

FuncActiv = Callable[[Tensor], Tensor]


class Layer:
    """
    A layer is a structure containing the weights of the neural network
    """

    def __init__(self, inodes: int, onodes: int, func: FuncActiv, deriv_func: FuncActiv) -> None:
        self.inodes: int = inodes
        self.onodes: int = onodes
        self.weights: Tensor = np.random.randn(inodes, onodes)
        self.bias: Tensor = np.random.randn(onodes)
        self.func = func
        self.deriv_func = deriv_func
        self.error: Any = None
        self.delta: Any = None
        self.inputs: Tensor = Tensor(0)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        The feed forward method.
        With Q = the activation function,
        It will return:
        Q(inputs @ weights + bias)
        """
        raise NotImplementedError


class Activation(Layer):
    """
    It's a subclass of Layer that defines the `forward` method
    """

    def __init__(
        self,
        inodes: int,
        onodes: int,
        func: FuncActiv,
        deriv_func: FuncActiv
    ) -> None:
        super().__init__(inodes, onodes, func, deriv_func)

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = self.func(inputs @ self.weights + self.bias)
        return self.inputs


def relu(x: Tensor) -> Tensor:
    return np.maximum(0, x)


def deriv_relu(x: Tensor) -> Tensor:
    return x


class Relu(Activation):
    """
    Layer with the relu activation function
    """

    def __init__(self, inodes, onodes):
        super().__init__(inodes, onodes, relu, deriv_relu)


def sigmoid(x: Tensor) -> Tensor:
    return 1 / (1+np.exp(-x))


def deriv_sigmoid(x: Tensor) -> Tensor:
    return x * (1-x)  # type: ignore


class Sigmoid(Activation):
    """
    Layer with the sigmoid activation function
    """

    def __init__(self, inodes, onodes):
        super().__init__(inodes, onodes, sigmoid, deriv_sigmoid)


def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)


def deriv_tanh(x: Tensor) -> Tensor:
    return 1 - tanh(x) ** 2


class Tanh(Activation):
    """
    Layer with the tanh activation function
    """

    def __init__(self, inodes, onodes):
        super().__init__(inodes, onodes, tanh, deriv_tanh)
