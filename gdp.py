
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import numpy as np
from keras.models import load_model

from data import load_from_H5
from bnn import bnn
from viz import plot_predictions3



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

num_hidden_layers = 5
# num hidden units
n_hidden = 1024
n_epochs = 1000
epochs_multiplier = 1
epochs_multiplier
best_tau = 0.1
best_dropout = 0.1
net = bnn(
    X_train,
    y_train,
    ([int(n_hidden)] * num_hidden_layers),
    normalize=False,
    tau=best_tau,
    dropout=best_dropout
)

print("Training a new model...")
net.train(X_train, y_train, n_epochs=n_epochs, batch_size=128, verbose=1)
net.model.save("models/gdp.h5")

plot_predictions3(net, (X_train, y_train), X, dates, sample_inds, out_of_sample_inds, iters=10000, n_std=4)
plt.show()
