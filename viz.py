import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras


def plot_predictions_only(net, trainset, X_test,
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


def plot_predictions(net, trainset, X_test,
                     iters=200, n_std=2, ax=None,
                     zoomed=False):
    X_train, y_train = trainset

    if ax is None:
        plt.close("all")
        plt.clf()
        fig, ax = plt.subplots(1, 1)
        if zoomed:
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


def plot_predictions_no_legend(net, trainset, X_test,
                               iters=200, n_std=2, ax=None,
                               zoomed=False):
    X_train, y_train = trainset

    if ax is None:
        plt.close("all")
        plt.clf()
        fig, ax = plt.subplots(1, 1)
        if zoomed:
            plt.axis([-1.75, 3.75, -20, 20])
    Xs = np.concatenate((X_test, X_train))
    Xs = np.sort(Xs, axis=0)
    y_means, y_stds = net.predict(Xs, T=iters)
    ax.plot(X_train, y_train, "r", alpha=0.8, 
            # label="observed"
            )
    ax.plot(Xs, y_means,
            # label="prediction",
            color="k",
            linestyle=":",
            linewidth=.75,
            alpha=.8)
    for i in range(n_std):
        ax.fill_between(
            Xs.squeeze(),
            (y_means - y_stds * ((i+1)/2)).squeeze(),
            (y_means + y_stds * ((i+1)/2)).squeeze(),
            color="b",
            alpha=0.5**(i+1)
        )
    # ax.legend()
    return ax


def plot_activation_fn(activation, lo=-5, hi=5, step=0.1, sess=None, ax=None):
    if sess is None:
        sess = tf.Session()
    if ax is None:
        _, ax = plt.subplots(1, 1)
    x = np.arange(lo, hi, step)
    activation_fn = keras.activations.get(activation)
    y = activation_fn(x).eval(session=sess)
    ax.plot(x, y)
    ax.set(ylabel=activation)
    return ax

def plot_activation_fns(activations, lo=-5, hi=5, step=0.1, sess=None)
    fig, axes = plt.subplots(len(activations), sharex=True)
    fig.set_figwidth(4)
    fig.set_figheight(1.75 * len(activations))
    fig.tight_layout()
    for i in range(len(activations)):
        ax = axes[i]
        plot_activation_fn(activations[i], lo, hi, sess=sess, ax=ax)
    return fig
