"""
Module containing a range of model types and architectures commonly used in
    machine learning applications.
"""


import numpy as np
from numpy.random import uniform


# Multi-layer perceptron -----------------------------------------------------
class Perceptron():
    def __init__(self, layers=None, input_size=None):
        self.layers = layers if layers is not None else []
        self.weights = [np.array([[]]) for _ in self.layers]
        self.biases = [np.array([]) for _ in self.layers]
        self.input_size = input_size
        self.loss = None
        self.learning_rate = 1
        self.metric = None

        self._x = [] # x
        self._c = [] # w * x + b
        self._a = [] # a_func(_c)
        self._p = [] # a_func'(_c)

        self.verbose = True
        self.print_length = 0


    def add_layer(self, layer):
        self.layers.append(layer)
        inp = self.layers[-2].num_units if len(self.layers) > 1 else self.input_size

        # TODO initialization should depend on layer size
        w_l = np.array([[uniform(-1, 1) for _ in range(inp)] for _ in range(layer.num_units)])
        b_l = np.array([0] * layer.num_units)

        self.weights.append(w_l)
        self.biases.append(b_l)


    def compile(self, loss, lr):
        self.loss = loss
        self.learning_rate = lr


    # TODO predict batches
    def predict(self, x, save=False):
        for i, layer in enumerate(self.layers):
            c = np.dot(self.weights[i], x) + self.biases[i]
            a = layer.a_func(c)
            if save:
                p = layer.a_func(c, derivative=True)
                self._x.append(x)
                self._c.append(c)
                self._a.append(a)
                self._p.append(p)
            x = a
        return x


    def train(self, x, y, epochs=1, verbose=True):
        self.verbose = verbose
        self._print_clear()

        predictions = []
        print("Training --------------------------")
        for e in range(epochs):
            print "Epoch "+str(e+1)+":",
            for sample, target in zip(x, y):
                # perform forward pass
                predictions.append(self.predict(sample, save=True))

                # perform backward pass
                dL_dA = self.loss(predictions[-1], target, derivative=True)
                for i in range(-1, -(1+len(self.layers)), -1):
                    dA_dC = self._p[i]
                    dC_dW = self._x[i]
                    dL_dW = np.outer(dL_dA * dA_dC, dC_dW)
                    dL_dB = dL_dA * dA_dC * 1

                    self.weights[i] = self.weights[i] - self.learning_rate * dL_dW
                    self.biases[i] = self.biases[i] - self.learning_rate * dL_dB
                    
                    dL_dA = np.dot(dL_dA * dA_dC, self.weights[i])

                self._print_training(predictions, y)
            self._print_clear()


    def _print_clear(self):
        if not self.verbose: return
        self.print_length = 0
        print('')

    
    def _print_training(self, predictions, targets):
        if not self.verbose: return
        delete = "\b" * self.print_length

        index = len(predictions) - 1
        loss = "Loss={:.3f}".format(self.loss(predictions[-1], targets[index]))
        progress = ", Progress={:.2f}%".format((float)(len(predictions)) / len(targets) * 100)
        out = delete + loss + progress
        self.print_length = len(out)+2

        print out,

        # if self.metric is not None:
        #     idx_predictions = np.array([list(p).index(p.max()) for p in predictions])
        #     idx_targets = np.array([list(t).index(t.max()) for t in targets])
        #     current_accuracy = self.metric(idx_predictions, idx_targets)
        #     out = out + ", {}={:.2f}".format(self.metric.__name__, current_accuracy)


# Nearest-neighbor classifier ------------------------------------------------
class NNclassifier():
    pass
