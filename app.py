from bbnet.neuralnetwork import NeuralNetwork
from bbnet.layer import Sigmoid, Tanh
from bbnet.tensor import Tensor


def main():
    nn = NeuralNetwork([Tanh(2, 4), Sigmoid(4, 4), Sigmoid(4, 1)])
    inputs = Tensor.from_array([[0, 1], [1, 0], [0, 0], [1, 1]])
    targets = Tensor.from_array([[1], [1], [0], [0]])

    nn.train(inputs, targets, epochs=4000, log_each=100)
    for i, t in zip(inputs, targets):
        t = t.tolist()
        print(f'Expected {t[0]}  guessed {nn.guess(i).tolist()[0]:.2}')


if __name__ == "__main__":
    main()
