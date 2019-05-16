import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import argparse
from keras.models import load_model
from tqdm.auto import tqdm
import os
import pickle
import pandas as pd
import numpy as np
import json
import keras.backend as K

from data import load_from_H5
from bnn import bnn
from viz import plot_predictions_gold

def gaussian(x):
    return K.exp(-K.pow(x,2))

def stats_row(epoch, rmse_standard_pred=np.nan, rmse=np.nan, test_ll=np.nan,
              all_std=np.nan, train_std=np.nan, runtime=np.nan):
    return pd.DataFrame.from_dict({
            epoch: {
                "rmse_standard_pred": rmse_standard_pred,
                "rmse": rmse,
                "log_loss": test_ll,
                "all_std": all_std,
                "train_std": train_std,
                "runtime": runtime\
            }
    }, orient='index')


parser = argparse.ArgumentParser(description='Mauna Loa experiment runner')
parser.add_argument('-e', '--epochs', default=10000, type=int, help="Number of epochs")
parser.add_argument('-a', '--activations', default=None, nargs='+', help='Activations for this experiment')
parser.add_argument('-o', '--overwrite', action='store_true', help='Overwrite existing experiment?')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

test_hdf5_filepath = 'data/Mauna Loa/test.h5'
train_hdf5_filepath = 'data/Mauna Loa/train.h5'

testset = load_from_H5(test_hdf5_filepath)
trainset = load_from_H5(train_hdf5_filepath)

X_test, y_test = testset
X_train, y_train = trainset

num_hidden_layers = 1
#n_hidden = 1024  # num hidden units
normalize = False
epochs = args.epochs
epoch_step_size = 200
batch_size = 128

test_iters = 1000 # low to speed up training
n_std = 2

weight_prior = 'glorot_uniform'
bias_prior = 'zeros'

tau = 0.1 #10
lengthscale = 1e-2  # 5
optimizer = 'adam'
dropout = 0.1


activations = args.activations if args.activations else \
    [   gaussian
        #'relu',
        #'tanh',
        #'sigmoid',
        # 'linear',
        # 'softplus',
        # 'elu',
        # 'softmax',
        # 'exponential'
    ]
nums = [10000]#[100, 256, 512, 1024] #[100, 256, 512] # 1024, 2048, 4096]
#activations.reverse()

num_nodes_dir = "experiments/mauna_loa/num_nodes/"

# experiments
stats = {}
for a in tqdm(range(len(activations))):
    for num in nums:
        activation = activations[a]
        experiment_dir = "{}{}{}{}/".format(num_nodes_dir, activation, num_hidden_layers, num)

        if activation == gaussian:
            name =  "gaussian"
        else:
            name = activation
        plot_dir = experiment_dir + "plots/"
        stats_file = experiment_dir + "stats.csv"
        model_file = experiment_dir + "model.h5"

        # create dirs
        for d in [experiment_dir, plot_dir]:
            if not os.path.exists(d):
                os.makedirs(d)

        print("Starting the {}{} experiment...".format(activation, num))

        # find model and pick up where left off
        if os.path.exists(plot_dir) and os.path.exists(stats_file) and \
                 os.path.exists(model_file) and not args.overwrite:
            stats = pd.read_csv(stats_file, index_col=0)
            # start at max recorded epoch
            cur_epoch = stats.index.max()
            print("Existing {} experiment found! Resuming at epoch {}.".format(name, cur_epoch))
            net = bnn(
                X_train,
                y_train,
                ([int(num)] * num_hidden_layers),
                normalize=normalize,
                tau=tau,
                dropout=dropout,
                activation=activation,
                weight_prior=weight_prior,
                bias_prior=bias_prior,
                model=load_model(model_file)
            )
        else:
            print("Overwriting existing {} experiment".format(name) if args.overwrite \
                      else "No existing {}experiment found.".format(name))
            # otherwise start from scratch


            cur_epoch = 0
            stats = stats_row(cur_epoch)
            stats.to_csv(stats_file) # initialize
            net = bnn(
                X_train,
                y_train,
                ([int(num)] * num_hidden_layers),
                normalize=normalize,
                tau=tau,
                dropout=dropout,
                activation=activation,
                weight_prior=weight_prior,
                bias_prior=bias_prior,
                model=None
            )

            # save architecture
            net.model.save(model_file)

        stats.index.name = 'epochs'
        if epoch_step_size > epochs:
            epoch_step_size = epochs
        end_epoch = cur_epoch + epochs
        while cur_epoch < end_epoch:
            cur_epoch += epoch_step_size

            print("Training model with {} through epoch {}...".format(name, cur_epoch))
            net.train(X_train, y_train, epochs=epoch_step_size, batch_size=batch_size, verbose=0)

            net.model.save(model_file)
            # plot
            fig = plt.figure()
            plt.xlim(-1.75, 3.75)

            Xs = np.concatenate((X_test, X_train))
            Xs = np.sort(Xs, axis=0)
            y_means, y_stds = net.predict(Xs, T=test_iters)
            plot_predictions_gold(X_train, Xs, y_train, y_means, y_stds, n_std)

            # save plots
            figname = "{}{}_epoch_{}".format(plot_dir, name, cur_epoch)
            plt.savefig("{}.png".format(figname))
            # save reloadable/editable plot
            # with open("{}.fig.pickle".format(figname), "wb") as f:
            #     pickle.dump(fig, f)
            plt.close()

            # record test stats
            all_std = y_stds.mean()
            train_std = y_stds[:X_test.shape[0]].mean()
            rmse_standard_pred, rmse, test_ll = net.test(X_test, y_test, T=test_iters)
            stats = stats.append(stats_row(cur_epoch, rmse_standard_pred, rmse, test_ll,
                                           all_std, train_std, net.running_time))
            stats.to_csv(stats_file)
