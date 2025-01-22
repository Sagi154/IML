import numpy as np
from scipy.special import softmax, logsumexp

class Network(object):
    
    def __init__(self, sizes):
        """
        The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        is [784, 40, 10] then it would be a three-layer network, with the
        first layer (the input layer) containing 784 neurons, the second layer 40 neurons,
        and the third layer (the output layer) 10 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution centered around 0.
        """
        self.num_layers = len(sizes) - 1
        self.sizes = sizes
        self.parameters = {}
        for l in range(1, len(sizes)):
            self.parameters['W' + str(l)] = np.random.randn(sizes[l], sizes[l-1]) * np.sqrt(2. / sizes[l-1])
            self.parameters['b' + str(l)] = np.zeros((sizes[l], 1))


    def relu(self,x):
        """TODO: Implement the relu function."""
        return np.maximum(0,x)

    def relu_derivative(self,x):
        """TODO: Implement the derivative of the relu function."""
        r_dev = np.zeros_like(x, dtype=int)
        r_dev[x > 0] = 1
        return r_dev

    def cross_entropy_loss(self, logits, y_true):
        m = y_true.shape[0]
        # Compute log-sum-exp across each column for normalization
        log_probs = logits - logsumexp(logits, axis=0)
        y_one_hot = np.eye(10)[y_true].T  # Assuming 10 classes
        # Compute the cross-entropy loss
        loss = -np.sum(y_one_hot * log_probs) / m
        return loss

    def cross_entropy_derivative(self, logits, y_true):
        """ Input: "logits": numpy array of shape (10, batch_size) where each column is the network output on the given example (before softmax)
                    "y_true": numpy array of shape (batch_size,) containing the true labels of the batch
            Returns: a numpy array of shape (10,batch_size) where each column is the gradient of the loss with respect to y_pred (the output of the network before the softmax layer) for the given example.
        """
        softmax_v = softmax(logits, axis=0)

        one_hot_y = np.eye(10)[y_true].T

        return (softmax_v - one_hot_y)

        


    def forward_propagation(self, X):
        """Implement the forward step of the backpropagation algorithm.
            Input: "X" - numpy array of shape (784, batch_size) - the input to the network
            Returns: "ZL" - numpy array of shape (10, batch_size), the output of the network on the input X (before the softmax layer)
                    "forward_outputs" - A list of length self.num_layers containing the forward computation (parameters & output of each layer).
        """
        forward_outputs = []
        layers = self.num_layers

        prev_Z = X
        for l in range(1, layers + 1):
            W = self.parameters["W" + str(l)]
            b = self.parameters["b" + str(l)]
            VL = np.dot(W, prev_Z) + b
            if l < layers:
                ZL = self.relu(VL)
            else:
                ZL = VL
            forward_outputs.append([W, b, prev_Z, VL, ZL])
            prev_Z = ZL
            
        return ZL, forward_outputs

    def backpropagation(self, ZL, Y, forward_outputs):
        """Implement the backward step of the backpropagation algorithm.
            Input: "ZL" -  numpy array of shape (10, batch_size), the output of the network on the input X (before the softmax layer)
                    "Y" - numpy array of shape (batch_size,) containing the labels of each example in the current batch.
                    "forward_outputs" - list of length self.num_layers given by the output of the forward function
            Returns: "grads" - dictionary containing the gradients of the loss with respect to the network parameters across the batch.
                                grads["dW" + str(l)] is a numpy array of shape (sizes[l], sizes[l-1]),
                                grads["db" + str(l)] is a numpy array of shape (sizes[l],1).
        
        """
        grads = {}
        batch_size = Y.shape[0]
        delta_L = self.cross_entropy_derivative(ZL, Y)
        layers = self.num_layers

        grads["dW" + str(layers)] = np.dot(delta_L, forward_outputs[layers - 1][2].T) / batch_size
        grads["db" + str(layers)] = np.mean(delta_L, axis=1, keepdims=True)

        for l in range(layers - 1, 0, -1):
            delta = np.dot(self.parameters["W" + str(l + 1)].T, delta_L)
            x = delta * self.relu_derivative(forward_outputs[l - 1][3])
            grads["dW" + str(l)] = np.dot(x, forward_outputs[l - 1][2].T) / batch_size
            grads["db" + str(l)] = np.mean(x, axis=1, keepdims=True)
            delta_L = x

        return grads


    def sgd_step(self, grads, learning_rate):
        """
        Updates the network parameters via SGD with the given gradients and learning rate.
        """
        parameters = self.parameters
        L = self.num_layers
        for l in range(L):
            parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]
        return parameters

    def train(self, x_train, y_train, epochs, batch_size, learning_rate, x_test, y_test):
        epoch_train_cost = []
        epoch_test_cost = []
        epoch_train_acc = []
        epoch_test_acc = []
        for epoch in range(epochs):
            costs = []
            acc = []
            for i in range(0, x_train.shape[1], batch_size):
                X_batch = x_train[:, i:i+batch_size]
                Y_batch = y_train[i:i+batch_size]

                ZL, caches = self.forward_propagation(X_batch)
                cost = self.cross_entropy_loss(ZL, Y_batch)
                costs.append(cost)
                grads = self.backpropagation(ZL, Y_batch, caches)

                self.parameters = self.sgd_step(grads, learning_rate)

                preds = np.argmax(ZL, axis=0)
                train_acc = self.calculate_accuracy(preds, Y_batch, batch_size)
                acc.append(train_acc)

            average_train_cost = np.mean(costs)
            average_train_acc = np.mean(acc)
            print(f"Epoch: {epoch + 1}, Training loss: {average_train_cost:.20f}, Training accuracy: {average_train_acc:.20f}")

            epoch_train_cost.append(average_train_cost)
            epoch_train_acc.append(average_train_acc)

            # Evaluate test error
            ZL, caches = self.forward_propagation(x_test)
            test_cost = self.cross_entropy_loss(ZL, y_test)
            preds = np.argmax(ZL, axis=0)
            test_acc = self.calculate_accuracy(preds, y_test, len(y_test))
            # print(f"Epoch: {epoch + 1}, Test loss: {test_cost:.20f}, Test accuracy: {test_acc:.20f}")

            epoch_test_cost.append(test_cost)
            epoch_test_acc.append(test_acc)

        return self.parameters, epoch_train_cost, epoch_test_cost, epoch_train_acc, epoch_test_acc


    def calculate_accuracy(self, y_pred, y_true, batch_size):
      """Returns the average accuracy of the prediction over the batch """
      return np.sum(y_pred == y_true) / batch_size
