#################################
# Your name:
#################################


import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels



def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """
    n = data.shape[0]
    w_t = np.zeros(data.shape[1])
    for t in range(1, T + 1):
        i = np.random.randint(low=0, high=n)
        x_i = data[i]
        y_i = labels[i]
        eta_t = eta_0 / t
        if (y_i * ( w_t @ x_i )) < 1:
            w_t = (1 - eta_t) * w_t + (eta_t * C * y_i * x_i)
        else:
            w_t = (1 - eta_t) * w_t
    return w_t



#################################

# Place for additional code

def test_accuracy(data, labels, w):
    correct_count = 0
    n = data.shape[0]
    for x_i, y_i in zip(data, labels):
        prediction = np.sign(w @ x_i)
        correct_count += prediction == y_i
    return correct_count / n


def find_optimal_eta():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
    resolution = 18
    etas = np.logspace(-5, 3, num=resolution)
    avg_accuracies = np.empty(resolution)
    best_eta = 0
    best_accuracy = 0
    for index, eta_0 in enumerate(etas):
        accuracy_sum = 0
        for i in range(10):
            w = SGD_hinge(train_data, train_labels, 1, eta_0, 1000)
            accuracy = test_accuracy(validation_data, validation_labels, w)
            accuracy_sum += accuracy
        avg_accuracy = accuracy_sum / 10
        # print(f"Curr accuracy: {avg_accuracy}")
        # print(f"Curr eta: {eta_0}")
        # print(f"Best eta: {best_eta}")
        # print(f"Best accuracy: {best_accuracy}")
        best_eta = eta_0 if avg_accuracy > best_accuracy else best_eta
        best_accuracy = max(best_accuracy, avg_accuracy)
        avg_accuracies[index] = avg_accuracy

    plt.title("Find best eta_0")
    plt.xlabel('eta_0')
    plt.ylabel('Average Accuracy')
    plt.plot(etas[:-1], avg_accuracies[:-1], linestyle='-', color='b')
    plt.xscale('log')
    plt.show()

    return best_eta


def find_optimal_c(best_eta):
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
    # Create the c values array from 10^-5 to 10^5
    resolution = 11
    c_vals = np.logspace(-5, 5, num=resolution)
    avg_accuracies = np.empty(resolution)
    eta_0 = best_eta
    best_c = 0
    best_accuracy = 0
    for index, c in enumerate(c_vals):
        accuracy_sum = 0
        for i in range(10):
            w = SGD_hinge(train_data, train_labels, c, eta_0, 1000)
            accuracy = test_accuracy(validation_data, validation_labels, w)
            accuracy_sum += accuracy
        avg_accuracy = accuracy_sum / 10
        # print(f"Curr accuracy: {avg_accuracy}")
        # print(f"Curr eta: {eta_0}")
        # print(f"Best eta: {best_eta}")
        # print(f"Best accuracy: {best_accuracy}")
        best_c = c if avg_accuracy > best_accuracy else best_c
        best_accuracy = max(best_accuracy, avg_accuracy)
        avg_accuracies[index] = avg_accuracy

    plt.title("Find best C")
    plt.xlabel('C')
    plt.ylabel('Average Accuracy')
    plt.plot(c_vals, avg_accuracies, linestyle='-', color='b')
    plt.xscale('log')
    plt.show()

    return best_c

def train_classifier(eta, c):
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
    eta_0 = eta
    C = c
    T = 2000
    w = SGD_hinge(train_data, train_labels, C, eta_0, 1000)
    image = np.reshape(w, (28, 28))
    plt.imshow(image, interpolation='nearest')
    plt.title('Visualizing w as an Image')
    plt.show()

    accuracy = test_accuracy(validation_data, validation_labels, w)
    return accuracy




if __name__ == '__main__':
    best_eta = find_optimal_eta()
    print(f"Best eta: {best_eta}")
    best_c = find_optimal_c(best_eta)
    print(f"Best C: {best_c}")
    best_accuracy = train_classifier(best_eta, best_c)
    print(f"Best Accuracy: {best_accuracy}")

#################################
