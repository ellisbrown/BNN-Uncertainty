import os
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import numpy as np
from data import load_from_H5
import pickle
from keras.models import load_model

from data import load_from_H5
from bnn import bnn
from viz import plot_predictions



parser = argparse.ArgumentParser(description='Mauna Loa runner')
parser.add_argument('--trained_model', default='models/mauna_loa.h5',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--train', default=False, type=bool)
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

df = pd.read_csv('data/GDP/gdp_data.csv')
X = df[df.columns.difference(['GDPC1', 'yq'])].values
y = df['GDPC1'].values
dates = df['yq'].values
cutoff_date = 2006

sample_inds = np.squeeze(np.argwhere(dates<cutoff_date))
out_of_sample_inds = np.squeeze(np.argwhere(dates>=cutoff_date))

X_train = X[sample_inds, :]
X_test = X[out_of_sample_inds, :]
y_train = y[sample_inds]
y_test = y[out_of_sample_inds]
trainset = X_train, y_train



num_hidden_layers = 5
n_hidden = 1024 # num hidden units
epochs = 10000
batch_size = 128
epochs_multiplier = 1
tau = 0.1
dropout = 0.1
normalize = True
activations = ['relu'] #['linear', 'relu', 'tanh', 'sigmoid', 'exponential']

for activation in activations:
    experiment_dir = "experiments/gdp/0/{}/".format(activation)

    net = bnn(
        X_train,
        y_train,
        ([int(n_hidden)] * num_hidden_layers),
        normalize=normalize,
        tau=tau,
        dropout=dropout,
        activation=activation
    )

    net.train(X_train, y_train, epochs=epochs, batch_size=batch_size,
              verbose=1)

    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    net.model.save("{}model_{}_.h5".format(experiment_dir, cutoff_date))

    plot_predictions(net, trainset, X_test, y_test, X, dates, sample_inds, out_of_sample_inds, iters=1000, n_std=4)
    plt.axvline(x=cutoff_date)
    plt.axvline(x=2008)
    #plt.plot(y_test)

    #load_from_H5("experiments/gdp/0/relu/model_2006_.h5")

    plt.savefig("{}plot_{}_{}.png".format(experiment_dir, cutoff_date, activation))
