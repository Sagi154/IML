#################################
# Your name: Sagi Eisenberg
#################################

import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import svm, datasets


def plot_results(models, titles, X, y, plot_sv=False):
    # # Calculate number of rows and columns dynamically
    # num_plots = len(titles)
    # ncols = 3  # Max number of plots per row (you can adjust this number)
    # nrows = (num_plots // ncols) + (num_plots % ncols > 0)  # Calculate rows needed
    #
    # # Set the figure size based on the number of rows and columns
    # fig_width_per_plot = 5  # Width of each subplot
    # fig_height = 5  # Height of each subplot
    # figsize = (fig_width_per_plot * ncols, fig_height * nrows)
    #
    # # Update plt.subplots to use the calculated nrows and ncols
    # fig, sub = plt.subplots(nrows, ncols, figsize=figsize)

    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(1, len(titles))  # 1, len(list(models)))

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    if len(titles) == 1:
        sub = [sub]
    else:
        sub = sub.flatten()
    for clf, title, ax in zip(models, titles, sub):
        # print(title)
        plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        if plot_sv:
            sv = clf.support_vectors_
            ax.scatter(sv[:, 0], sv[:, 1], c='k', s=60)

        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title, fontsize=9)
        ax.set_aspect('equal', 'box')
    fig.tight_layout()
    plt.show()

def make_meshgrid(x, y, h=0.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out



C = 10
n = 100


# Data is labeled by a circle

radius = np.hstack([np.random.random(n), np.random.random(n) + 1.5])
angles = 2 * math.pi * np.random.random(2 * n)
X1 = (radius * np.cos(angles)).reshape((2 * n, 1))
X2 = (radius * np.sin(angles)).reshape((2 * n, 1))

X = np.concatenate([X1,X2],axis=1)
y = np.concatenate([np.ones((n,1)), -np.ones((n,1))], axis=0).reshape([-1])


############ Question a ####################

def ques_a():
    linear_model = (svm.SVC(kernel='linear', C=C))
    poly2_model = (svm.SVC(kernel='poly', C=C, coef0=0, degree=2))
    poly3_model = (svm.SVC(kernel='poly', C=C, coef0=0, degree=3))

    models_estimators = [linear_model.fit(X, y), poly2_model.fit(X, y), poly3_model.fit(X, y)]

    models_names = ['Linear Kernel', 'Polynomial (degree 2) Kernel',
                    'Polynomial (degree 3) Kernel']

    plot_results(models_estimators, models_names, X, y)

############ Question b ####################

def ques_b():
    linear_model = (svm.SVC(kernel='linear', C=C))
    non_homo_poly2_model = (svm.SVC(kernel='poly', C=C, coef0=1, degree=2))
    non_homo_poly3_model = (svm.SVC(kernel='poly', C=C, coef0=1, degree=3))
    models_estimators = [linear_model.fit(X, y), non_homo_poly2_model.fit(X, y), non_homo_poly3_model.fit(X, y)]

    models_names = ['Linear Kernel', 'Non_homo_polynomial (degree 2) Kernel',
                    'Non_homo_polynomial (degree 3) Kernel']

    plot_results(models_estimators, models_names, X, y)

############ Question c ####################

def alter_label(label):
    if label < 0 and np.random.rand() < 0.1:
        label = 1
    return label

def ques_c():
    new_y = np.vectorize(alter_label)(y)

    poly2_model = (svm.SVC(kernel='poly', C=C, coef0=1, degree=2))
    rbf_model = (svm.SVC(kernel= 'rbf' , C=C, gamma=10))

    models_names = ['Polynomial (degree 2) Kernel', 'RBF kernel']

    models_estimators = [poly2_model.fit(X, new_y), rbf_model.fit(X, new_y)]

    plot_results(models_estimators, models_names, X, new_y)

def c_gamma_check():
    new_y = np.vectorize(alter_label)(y)
    resolution = 6
    gamma_vals = np.logspace(-2, 3, num=resolution)
    models_estimators = []
    models_names = []
    for gamma in gamma_vals:
        models_estimators.append(svm.SVC(kernel= 'rbf' , C=C, gamma=gamma).fit(X, new_y))
        models_names.append(f"RBF kernel (gamma={gamma})")
    plot_results(models_estimators, models_names, X, new_y)



if __name__ == '__main__':
    ques_a()
    # ques_b()
    # ques_c()
    # c_gamma_check()

















