import os
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import numpy as np
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

sample_inds = np.squeeze(np.argwhere(dates<2008))
out_of_sample_inds = np.squeeze(np.argwhere(dates>=2008))

X_train = X[sample_inds, :]
X_test = X[out_of_sample_inds, :]
y_train = y[sample_inds]
y_test = y[out_of_sample_inds]
trainset = X_train, y_train



num_hidden_layers = 5
n_hidden = 1024 # num hidden units
epochs = 500
batch_size = 128
epochs_multiplier = 1
tau = 0.1
dropout = 0.1
normalize = False
activations = ['linear', 'relu', 'tanh', 'sigmoid', 'exponential']

experiment_dir = "experiments/gdp/0/relu/"

net = bnn(
    X_train,
    y_train,
    ([int(n_hidden)] * num_hidden_layers),
    normalize=normalize,
    tau=tau,
    dropout=dropout,
    activation='relu'
)

net.train(X_train, y_train, epochs=epochs, batch_size=batch_size,
          verbose=1)

if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)
net.model.save("{}model.h5".format(experiment_dir))

plot_predictions(net, trainset, X_test, X, dates, sample_inds, out_of_sample_inds, iters=1000, n_std=4)
plt.savefig("{}plot.png".format(experiment_dir))
