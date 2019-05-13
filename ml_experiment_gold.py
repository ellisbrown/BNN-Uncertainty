
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import argparse
from keras.models import load_model
from tqdm.auto import tqdm
import os
import pickle
import pandas as pd

from data import load_from_H5
from bnn import bnn
from viz import plot_predictions

parser = argparse.ArgumentParser(description='Mauna Loa experiment runner')
parser.add_argument('-x', '--exp', default=0, type=int, help='Experiment number')
parser.add_argument('-e', '--epochs', default=50, type=int, help="Number of epochs")
parser.add_argument('-a', '--activations', default=None, nargs='+', help='Activations for this experiment')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

exp_num = args.exp

test_hdf5_filepath = 'data/Mauna Loa/test.h5'
train_hdf5_filepath = 'data/Mauna Loa/train.h5'

testset = load_from_H5(test_hdf5_filepath)
trainset = load_from_H5(train_hdf5_filepath)

X_test, y_test = testset
X_train, y_train = trainset

num_hidden_layers = 5
n_hidden = 1024  # num hidden units
epochs = args.epochs
batch_size = 128
tau = 10
lengthscale = 1e-2  # 5
optimizer = 'adam'
dropout = 0.1
normalize = False
test_iters = 1000
weight_prior='glorot_uniform'
bias_prior='zeros'

activations = args.activations if args.activations else \
    [
        'relu',
        'tanh',
        'sigmoid',
        'linear',
        'softplus',
        'elu',
        'softmax',
        'exponential'
    ]


experiment_dir = "experiments/mauna_loa/gold/"

# autoincrement exp number
exp_num = args.exp if args.exp > 0 else \
    int(max([name for name in os.listdir(experiment_dir)
             if os.path.isdir(os.path.join(experiment_dir, name)) and name[0] != "."])) + 1

experiment_dir = experiment_dir + "{}/".format(exp_num)
if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)
experiment_dir = "{}exp_{}_".format(experiment_dir, exp_num)

# experiments
stats = {}
for a in tqdm(range(len(activations))):
    activation = activations[a]
    print("Starting the {} experiment...".format(activation))
    net = bnn(
        X_train,
        y_train,
        ([int(n_hidden)] * num_hidden_layers),
        normalize=normalize,
        tau=tau,
        dropout=dropout,
        activation=activation,
        weight_prior=weight_prior,
        bias_prior=bias_prior
    )

    print("Training model with {}...".format(activation))
    net.train(X_train, y_train, epochs=epochs, batch_size=batch_size,
              verbose=1)

    net.model.save("{}{}_model.h5".format(experiment_dir, activation))

    # record test stats
    rmse_standard_pred, rmse, test_ll = net.test(X_test, y_test, T=test_iters)
    stats[activation] = {
        "rmse_standard_pred": rmse_standard_pred,
        "rmse": rmse,
        "log_loss": test_ll}

    fig = plt.figure()
    plt.xlim(-1.75, 3.75)
    # create prediction plot
    plot_predictions(net, trainset, X_test, iters=test_iters, n_std=2)

    # save plots
    figname = "{}{}_plot".format(experiment_dir, activation)
    plt.savefig("{}.png".format(figname))
    # save reloadable/editable plot
    with open("{}.fig.pickle".format(figname), "wb") as f:
        pickle.dump(fig, f)
    plt.close()

# save stats
df = pd.DataFrame.from_dict(stats, orient='index')
df.to_csv("{}stats.csv".format(experiment_dir))

