import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import numpy as np
import keras
from keras.models import load_model
import tqdm
import os

from data import load_from_H5
from bnn import bnn

# set tf debug off
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

parser = argparse.ArgumentParser(description='Quartile plot runner')
parser.add_argument('-a', '--activations', nargs='+', default=[
    'relu',
    'tanh',
    'sigmoid',
    'linear',
    'softplus',
    'elu',
    'softmax',
    'softsign',
    # 'exponential',
    'hard_sigmoid'
], help='Activation for this experiment')
parser.add_argument('-x', '--experiments', nargs='+', default=[
    # "default",
    "glorot_normal_prior",
    # "random_normal_prior"
], help="Name of experiment to run")
parser.add_argument('-o', '--overwrite', action='store_true', help='Overwrite existing experiment?')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

test_hdf5_filepath = 'data/Mauna Loa/test.h5'
train_hdf5_filepath = 'data/Mauna Loa/train.h5'

testset = load_from_H5(test_hdf5_filepath)
trainset = load_from_H5(train_hdf5_filepath)

X_test, y_test = testset
X_train, y_train = trainset
Xs = np.concatenate((X_test, X_train))
Xs = np.sort(Xs, axis=0)

num_hidden_layers = 5
n_hidden = 1024  # num hidden units
normalize = False
batch_size = 128

n_std = 2

tau = .10  # 10
lengthscale = 1e-2  # 5
optimizer = 'adam'
dropout = 0.1


def get_net(activation, experiment='default'):
    weight_prior = 'glorot_uniform'
    bias_prior = 'zeros'
    if experiment == 'glorot_normal_prior':
        weight_prior = bias_prior = 'glorot_normal'
    elif experiment == 'random_normal_prior':
        weight_prior = bias_prior = 'RandomNormal'
    gold_dir = "experiments/mauna_loa/gold/"
    experiment_dir = "{}{}/{}/".format(gold_dir, experiment, activation)
    model_file = experiment_dir + "model.h5"
    print("Loading {} net with {} activation...".format(experiment, activation))
    net = bnn(
            X_train,
            y_train,
            ([int(n_hidden)] * num_hidden_layers),
            normalize=normalize,
            tau=tau,
            dropout=dropout,
            activation=activation,
            weight_prior=weight_prior,
            bias_prior=bias_prior,
            model=load_model(model_file)
        )
    return net


def plot_qtl(activation, experiment='default', iters=20):

    experiment_dir = "experiments/qtl_plots/"
    figname = "{}{}_{}.png".format(experiment_dir, activation, experiment)

    if os.path.isfile(figname) and not args.overwrite:
        print("{} net with {} activation already exists, skipping...".format(experiment, activation))
        # skip
        return

    weight_prior = 'glorot_uniform'
    bias_prior = 'zeros'

    experiment_name = experiment
    if experiment_name == 'glorot_normal_prior':
        weight_prior = bias_prior = 'glorot_normal'
    elif experiment_name == 'random_normal_prior':
        weight_prior = bias_prior = 'RandomNormal'

    net = get_net(activation, experiment)
    print("Generating {} predictions...".format(iters))
    pbar = tqdm.trange(iters)
    Yt_hat = np.array(
            [net.model.predict(Xs, batch_size=500, verbose=0) for _ in pbar])
    qtl_1 = np.stack([np.percentile(Yt_hat[:, i], 25) for i in range(Yt_hat.shape[1])])
    means = np.stack([Yt_hat[:, i].mean() for i in range(Yt_hat.shape[1])])
    qtl_3 = np.stack([np.percentile(Yt_hat[:, i], 75) for i in range(Yt_hat.shape[1])])
    plt.close()
    plt.plot(X_train, y_train, "r", alpha=0.8, label="observed")
    plt.plot(Xs.squeeze(), means,
             label="prediction",
             color="k",
             linestyle=":",
             linewidth=.5,
             alpha=.8)
    plt.fill_between(
        Xs.squeeze(),
        qtl_1.squeeze(),
        qtl_3.squeeze(),
        color="b",
        alpha=0.25
    )
    plt.plot(Xs, means,
        label="mean",
        color="k",
        # linestyle=":",
        linewidth=1,
        alpha=.8)

    if os.path.isfile(figname):
        # overwrite
        os.remove(figname)  # Opt.: os.system("rm "+strFile)
    plt.savefig(figname)
    plt.close()
    return plt


test_iters = 100

def main():
    activations = args.activations
    experiments = args.experiments
    # activations.reverse()
    # experiments.reverse()
    for experiment in tqdm.tqdm(experiments):
        for activation in tqdm.tqdm(activations):
            plot_qtl(activation, experiment, iters=test_iters)


if __name__ == '__main__':
    main()
