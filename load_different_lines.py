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
import csv


from data import load_from_H5
from bnn import bnn
from viz import plot_predictions_gold

test_hdf5_filepath = 'data/Mauna Loa/test.h5'
train_hdf5_filepath = 'data/Mauna Loa/train.h5'

testset = load_from_H5(test_hdf5_filepath)
trainset = load_from_H5(train_hdf5_filepath)

X_test, y_test = testset
X_train, y_train = trainset

num_hidden_layers = 5
#n_hidden = 1024  # num hidden units
normalize = False
batch_size = 128

test_iters = 1000 # low to speed up training
n_std = 1

weight_prior = 'glorot_uniform'
bias_prior = 'zeros'

tau = 10
lengthscale = 1e-2  # 5
optimizer = 'adam'
dropout = 0.1


activations = [
        'relu',
        #'tanh',
        #'sigmoid',
        #'linear',
        #'softplus',
        #'elu',
        #'softmax',
        # 'exponential'
    ]
nums = [100, 256, 512, 1024]#, 256, 512, 1024] #, 1024, 2048, 4096]
#activations.reverse()

plot_std = True
plot_name = "num_nodes_with_std"

num_nodes_dir = "experiments/mauna_loa/num_nodes/"
cur_epoch = 10000
fig = plt.figure()
shading_colors = ['b', 'g', 'r', 'c', 'm', 'y']

Xs = np.concatenate((X_test, X_train))
Xs = np.sort(Xs, axis=0)

y_means_gp = np.genfromtxt('/Users/Ethan/Google Drive/School/Fed/EECS 6699/Project/BNN-Uncertainty/y_means_gp.csv', delimiter=',')
y_stds_gp = np.genfromtxt('/Users/Ethan/Google Drive/School/Fed/EECS 6699/Project/BNN-Uncertainty/y_stds_gp.csv', delimiter=',')
print(y_means_gp)
print(Xs)
plt.plot(Xs, y_means_gp, c = 'k', label = "Gaussian Process")
plt.xlim(-2, 2)

if plot_std:
    for i in range(n_std):
        plt.fill_between(
            Xs.squeeze(),
            (y_means_gp - y_stds_gp * ((i+1)/2)).squeeze(),
            (y_means_gp + y_stds_gp * ((i+1)/2)).squeeze(),
            color='k',
            alpha=0.25**(i+1)
            )

for (shading_col_ind, num) in enumerate(nums):
    for activation in activations:
        experiment_dir = "{}{}{}/".format(num_nodes_dir, activation, num)
        plot_dir = experiment_dir + "plots/"
        stats_file = experiment_dir + "stats.csv"
        model_file = experiment_dir + "model.h5"

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
    plt.xlim(-1.75, 3.75)

    Xs = np.concatenate((X_test, X_train))
    Xs = np.sort(Xs, axis=0)
    y_means, y_stds = net.predict(Xs, T=test_iters)
    plt.plot(Xs, y_means, c = shading_colors[shading_col_ind], label = "Hidden Neurons = "+str(num))
    if plot_std:
        for i in range(n_std):
            plt.fill_between(
                Xs.squeeze(),
                (y_means - y_stds * ((i+1)/2)).squeeze(),
                (y_means + y_stds * ((i+1)/2)).squeeze(),
                color=shading_colors[shading_col_ind],
                alpha=0.25**(i+1)
                )

plt.gca().legend()
plt.savefig("{}.png".format(plot_name))
plt.close()
