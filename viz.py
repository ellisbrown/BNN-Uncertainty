import matplotlib.pyplot as plt
import numpy as np


def plot_predictions(net, trainset, X_test,
                     iters=200, n_std=2, ax=None):
    X_train, y_train = trainset

    if ax is None:
        plt.close("all")
        plt.clf()
        fig, ax = plt.subplots(1, 1)
        plt.axis([-1.75, 3.75, -20, 20])
    y_means, y_stds = net.predict(X_test, T=iters)
    ax.plot(X_train, y_train, color="r", alpha=0.8, label="observed")
    ax.plot(X_test, y_means, "k--", label="mean")
    for i in range(n_std):
        ax.fill_between(
            X_test.squeeze(),
            (y_means - y_stds * ((i+1)/2)).squeeze(),
            (y_means + y_stds * ((i+1)/2)).squeeze(),
            color="b",
            alpha=0.5**(i+1)
        )
    ax.legend()
    return ax


def plot_predictions2(net, trainset, X_test,
                      iters=200, n_std=2, ax=None):
    X_train, y_train = trainset

    if ax is None:
        plt.close("all")
        plt.clf()
        fig, ax = plt.subplots(1, 1)
        plt.axis([-1.75, 3.75, -20, 20])
    Xs = np.concatenate((X_test, X_train))
    Xs = np.sort(Xs, axis=0)
    y_means, y_stds = net.predict(Xs, T=iters)
    plt.plot(X_train, y_train, "r", alpha=0.8, label="observed")
    plt.plot(Xs, y_means,
             label="prediction",
             color="k",
             linestyle=":",
             linewidth=.5,
             alpha=.8)
    for i in range(n_std):
        ax.fill_between(
            Xs.squeeze(),
            (y_means - y_stds * ((i+1)/2)).squeeze(),
            (y_means + y_stds * ((i+1)/2)).squeeze(),
            color="b",
            alpha=0.5**(i+1)
        )
    ax.legend()
    return ax

def plot_predictions3(net, trainset, X_test, dates, iters=200, n_std=2, ax=None):
    X_train, y_train = trainset

    if ax is None:
        plt.close("all")
        plt.clf()
        fig, ax = plt.subplots(1, 1)
        plt.axis([1985, 2017, np.amin(trainset[1]), np.amax(trainset[1])])
    y_means, y_stds = net.predict(X_test, T=iters)
    ax.plot(X_train, y_train, color="r", alpha=0.8, label="observed")
    ax.plot(X_test, y_means, "k--", label="mean")
    for i in range(n_std):
        ax.fill_between(
            X_test.squeeze(),
            (y_means - y_stds * ((i+1)/2)).squeeze(),
            (y_means + y_stds * ((i+1)/2)).squeeze(),
            color="b",
            alpha=0.5**(i+1)
        )
    ax.legend()
    return ax
