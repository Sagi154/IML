import matplotlib.pyplot as plt
import numpy as np
from backprop_network import *
from backprop_data import *

def ques_b():
    # Loading Data
    np.random.seed(0)  # For reproducibility
    n_train = 10000
    n_test = 5000
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)

    # Training configuration
    epochs = 30
    batch_size = 10
    learning_rates = [0.001, 0.01, 0.1, 1, 10]

    epochs_arr = [i for i in range(1, epochs + 1)]
    train_acc_arr = []
    train_loss_arr = []
    test_acc_arr = []

    # Network configuration
    layer_dims = [784, 40, 10]

    for lr in learning_rates:
        net = Network(layer_dims)
        params, epoch_train_loss, epoch_test_cost, epoch_train_acc, epoch_test_acc = \
            net.train(x_train, y_train, epochs, batch_size, lr, x_test=x_test, y_test=y_test)
        train_acc_arr.append(epoch_train_acc)
        train_loss_arr.append(epoch_train_loss)
        test_acc_arr.append(epoch_test_acc)
    
    # Training Accuracy
    plt.figure(figsize=(14, 7))
    for i, learning_rate in enumerate(learning_rates):
        plt.plot(epochs_arr, train_acc_arr[i], label=f'Learning rate : {learning_rate}')
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy with Different Learning Rates')
    plt.legend()
    plt.show()

    # Training Loss
    plt.figure(figsize=(14, 7))
    for i, learning_rate in enumerate(learning_rates):
        plt.plot(epochs_arr, train_loss_arr[i], label=f'Learning rate : {learning_rate}')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Loss with Different Learning Rates')
    plt.legend()
    plt.show()

    # Test Accuracy
    plt.figure(figsize=(14, 7))
    for i, learning_rate in enumerate(learning_rates):
        plt.plot(epochs_arr, test_acc_arr[i], label=f'Learning rate : {learning_rate}')
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy with Different Learning Rates')
    plt.legend()
    plt.show()

def ques_c():
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

    params, epoch_train_loss, epoch_test_cost, epoch_train_acc, epoch_test_acc = \
            net.train(x_train, y_train, epochs, batch_size, learning_rate, x_test=x_test, y_test=y_test)
    final_test_acc = epoch_test_acc[-1]
    print(f"Test accuracy on test set in the final epoch: {final_test_acc:.2f}")

def ques_d():
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
    params, epoch_train_cost, epoch_test_cost, epoch_train_acc, epoch_test_acc = \
        net.train(x_train, y_train, epochs, batch_size, learning_rate, x_test=x_test, y_test=y_test)

    W = params["W1"]
    f, a = plt.subplots(2, 5, figsize=(12, 6))
    f.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(a.flat):
        img = ax.imshow(W[i].reshape(28, 28), cmap='viridis', interpolation='nearest')
        ax.axis('off')
        ax.set_title(f'Class Number : {i}')
    f.colorbar(img, ax=a.ravel().tolist(), shrink=0.5)
    plt.show()

if __name__ == "__main__":
    ques_b()
    ques_c()
    ques_d()