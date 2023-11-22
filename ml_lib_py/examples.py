"""
Module containing a range of example use cases of neural network applications
    using the modules of ml-lib-py.
"""


import os
from ml_lib_py.networks.model import Perceptron
from ml_lib_py.networks.layer import Dense, Sigmoid
# from networks.activations import sigmoid, softmax
from networks.loss import categorical_crossentropy
from evaluation import accuracy
from data import load_mnist


def mlp_classify_mnist():
    # define model architecture
    mlp = Perceptron(input_size=784)

    # mlp.add_layer(Dense(units=784, a_func=sigmoid)) # input
    # mlp.add_layer(Dense(units=128, a_func=sigmoid)) # hidden 1
    # mlp.add_layer(Dense(units=32, a_func=sigmoid)) # hidden 2
    # mlp.add_layer(Dense(units=10, a_func=softmax)) # output

    mlp.compile(loss=categorical_crossentropy, lr=0.01)

    # either load pretrained weights or train the model
    if not os.path.exists("parameters.npz"):
        x_train, y_train, x_test, y_test = load_mnist(split='both')

        # before training
        predictions = [mlp.predict(sample) for sample in x_test]
        print("Test set acurracy before training: {:.2f}%"
              .format(accuracy(predictions, y_test) * 100))

        mlp.train(x=x_train, y=y_train, epochs=3)
        mlp.save_weights()
    else:
        x_test, y_test = load_mnist(split='test')
        mlp.load_weights()

    # after training
    predictions = [mlp.predict(sample) for sample in x_test]
    print("Test set acurracy before training: {:.2f}%"
              .format(accuracy(predictions, y_test) * 100))


def mlp_regress_boston():
    pass
