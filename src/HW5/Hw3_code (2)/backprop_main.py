import matplotlib.pyplot as plt
import numpy as np
from backprop_network import *
from backprop_data import *


def plot_weights_as_images(W):
    f, a = plt.subplots(2, 5, figsize=(12, 6))
    f.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(a.flat):
        img = ax.imshow(W[i].reshape(28, 28), cmap='viridis', interpolation='nearest')
        ax.axis('off')
        ax.set_title(f'Class Number: {i}')
    f.colorbar(img, ax=a.ravel().tolist(), shrink=0.5)
    plt.show()


def ex_b():
    # Loading Data
    np.random.seed(0)  # For reproducibility
    n_train = 10000
    n_test = 5000
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)

    # Training configuration
    epochs = 30
    batch_size = 10
    learning_rates = [0.001, 0.01, 0.1, 1, 10]

    # Network configuration
    layer_dims = [784, 40, 10]

    train_accs = []
    train_losses = []
    test_accs = []
    epochs_array = [i for i in range(1, epochs+1)]

    for learning_rate in learning_rates:
        net = Network(layer_dims)
        parameters, epoch_train_cost, epoch_test_loss, epoch_train_acc, epoch_test_acc = \
            net.train(x_train, y_train, epochs, batch_size, learning_rate, x_test=x_test, y_test=y_test)

        train_accs.append(epoch_train_acc)
        train_losses.append(epoch_train_cost)
        test_accs.append(epoch_test_acc)

    #training accuracy
    plt.figure(figsize=(14, 7))
    for i, learning_rate in enumerate(learning_rates):
        plt.plot(epochs_array, train_accs[i], label=f'Learning rate = {learning_rate}')
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy for Different Learning Rates')
    plt.legend()
    plt.show()

    #training loss
    plt.figure(figsize=(14, 7))
    for i, learning_rate in enumerate(learning_rates):
        plt.plot(epochs_array, train_losses[i], label=f'Learning rate ={learning_rate}')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Loss for Different Learning Rates')
    plt.legend()
    plt.show()

    #test accuracy
    plt.figure(figsize=(14, 7))
    for i, learning_rate in enumerate(learning_rates):
        plt.plot(epochs_array, test_accs[i], label=f'Learning rate ={learning_rate}')
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy for Different Learning Rates')
    plt.legend()
    plt.show()


def ex_c():
    # Loading Data
    np.random.seed(0)  # For reproducibility
    n_train = 60000
    n_test = 10000
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)

    # Training configuration
    epochs = 30
    batch_size = 10
    learning_rate = 0.1


    # Network configuration
    layer_dims = [784, 40, 10]
    net = Network(layer_dims)

    parameters, epoch_train_cost, epoch_test_cost, epoch_train_acc, epoch_test_acc = \
            net.train(x_train, y_train, epochs, batch_size,learning_rate , x_test=x_test, y_test=y_test)
    final_test_acc = epoch_test_acc[-1]
    print(f"Final test accuracy with learning rate 0.1: {final_test_acc:.2f}")

def ex_d():
    # Loading Data
    np.random.seed(0)  # For reproducibility
    n_train = 60000
    n_test = 10000
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)

    # Training configuration
    epochs = 30
    batch_size = 10
    learning_rate = 0.1


    layer_dims = [784, 10]

    net = Network(layer_dims)
    parameters, epoch_train_cost, epoch_test_cost, epoch_train_acc, epoch_test_acc = \
        net.train(x_train, y_train, epochs, batch_size, learning_rate, x_test=x_test, y_test=y_test)

    W = parameters["W1"]
    plot_weights_as_images(W)


def ex_e():
    # Loading Data
    np.random.seed(0)  # For reproducibility
    n_train = 60000
    n_test = 10000
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)

    # Training configuration
    epochs = 16
    learning_rate = 0.1
    layer_dims = [784, 200, 40, 10]
    batch_size = 40
    epochs_array = [i for i in range(1, epochs + 1)]
    net = Network(layer_dims)

    parameters, epoch_train_cost, epoch_test_cost, epoch_train_acc, epoch_test_acc = \
        net.train(x_train, y_train, epochs, batch_size, learning_rate, x_test=x_test, y_test=y_test)

    # Plotting test accuracy
    plt.plot(epochs_array, epoch_test_acc, label=f'Test Acc Learning rate = {0.1}')
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy')
    plt.legend()
    plt.show()
    final_test_acc = epoch_test_acc[-1]
    print(f"Final test accuracy with learning rate 0.1: {final_test_acc:.3f}")

if __name__ == '__main__':

    ex_b()
    ex_c()
    ex_d()
    ex_e()










