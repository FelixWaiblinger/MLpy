"""
Module containing a range of evaluation metrics commonly used in neural
    networks and machine learning applications.
"""


def accuracy(predictions, targets):
    correct = 0
    for p, t in zip(predictions, targets):
        if p == t: correct += 1

    return (float)(correct) / len(predictions)