import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import numpy as np
import keras
import tqdm
import os

from data import load_from_H5
from bnn import bnn

parser = argparse.ArgumentParser(description='Prior experiment runner')
parser.add_argument('-a', '--activation', default=None, help='Activation for this experiment')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

X_train = np.arange(-2, 2.01, 0.01)
y_train = np.zeros_like(X_train.squeeze())
X_train = X_train[:, np.newaxis]

# net params
num_hidden_layers = 5
n_hidden = 1024  # num hidden units
tau = 0.1
dropout = 0  # off
normalize = True


def prior_plot(activation, prior, iters=20):
    y_preds = []
    for _ in tqdm.trange(iters):
        net = bnn(
            X_train,
            y_train,
            ([int(n_hidden)] * num_hidden_layers),
            normalize=normalize,
            tau=tau,
            dropout=dropout,
            activation=activation,
            weight_prior=prior,
            bias_prior=prior
        )

        y_pred = net.model.predict(X_train, verbose=0)
        y_preds.append(y_pred)

        # individual plot
        plt.plot(X_train, y_pred,
                 # label="prediction",
                 color="b",
                 linestyle=":",
                 linewidth=.75,
                 alpha=.8)

    y_preds = np.array(y_preds)
    means = y_preds.mean(axis=0)
    qtl_1 = np.percentile(y_preds, 25, axis=0)
    qtl_3 = np.percentile(y_preds, 75, axis=0)
    plt.fill_between(
        X_train.squeeze(),
        qtl_1.squeeze(),
        qtl_3.squeeze(),
        color="k",
        alpha=0.25
    )
    plt.plot(X_train, means,
             label="mean",
             color="k",
             #         linestyle=":",
             linewidth=1,
             alpha=.7)
    # plt.xlim(-2, 2)
    plt.axis([-2, 2, -2, 2])


def prior_comparison_experiment(priors, activation='relu', iters=20):
    experiment_dir = "experiments/priors/"
    for prior in priors:
        print("Starting the {} prior experiment...".format(prior))
        prior_plot(activation=activation, prior=prior, iters=iters)
        figname = "{}{}_{}.png".format(experiment_dir, activation, prior)
        if os.path.isfile(figname):
            os.remove(figname)  # Opt.: os.system("rm "+strFile)
        plt.savefig(figname)
        plt.close()


def main():
    priors = [
        "RandomNormal",
        "RandomUniform",
        # "TruncatedNormal",
        # "VarianceScaling",
        # "lecun_uniform",
        "glorot_normal",
        "glorot_uniform",
        # "he_normal",
        # "lecun_normal",
        # "he_uniform"
    ]
    activation = 'relu' if args.activation is None else args.activation
    iters = 40
    prior_comparison_experiment(priors=priors, activation=activation, iters=iters)


if __name__ == '__main__':
    main()
