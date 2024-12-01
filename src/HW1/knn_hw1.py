import numpy as np
import matplotlib.pyplot as plt
import numpy.random
from sklearn.datasets import fetch_openml

# Loading the dataset
mnist = fetch_openml("mnist_784", as_frame=False)
data = mnist["data"]
labels = mnist["target"]
# Define training and test set of images
idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]


def knn(train_images, labels, query_image, k):
    dist_arr = np.linalg.norm(train_images - query_image, axis=1)  # Get distances of all train images from query image
    sorted_indices = np.argsort(dist_arr)  # Get a sort of those distances
    sorted_labels = labels[sorted_indices]  # Match the sorting order of the labels to the train images
    values, counts = np.unique(sorted_labels[0:k], return_counts=True)
    predict_label = values[np.argmax(counts)]  # Get the majority label
    return predict_label


def run_knn(n, k):
    correct_predictions = 0
    for i, query_image in enumerate(test):
        predict_label = knn(train[:n], train_labels[:n], query_image, k)
        actual_label = test_labels[i]
        correct_predictions += actual_label == predict_label
    return (correct_predictions / len(test)) * 100
