
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
from keras.models import load_model
from tqdm.auto import tqdm
import os

from data import load_from_H5
from bnn import bnn
from viz import plot_predictions

parser = argparse.ArgumentParser(description='Mauna Loa experiment runner')
parser.add_argument('--exp', default=0, type=int, 
                    help='Experiment number')
parser.add_argument('-f', default=None, type=str, 
                    help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()


test_hdf5_filepath = 'data/Mauna Loa/test.h5'
train_hdf5_filepath = 'data/Mauna Loa/train.h5'

testset = load_from_H5(test_hdf5_filepath)
trainset = load_from_H5(train_hdf5_filepath)

X_test, y_test = testset
X_train, y_train = trainset

num_hidden_layers = 5
n_hidden = 1024 # num hidden units
epochs = 40
batch_size = 128
epochs_multiplier = 1
tau = 0.1
dropout = 0.1
normalize = False
activations = ['linear', 'relu', 'tanh', 'sigmoid', 'exponential']

for activation in tqdm(activations):
    print("Starting the {} experiment...".format(activation))
    experiment_dir = "experiments/mauna_loa/{}/{}/".format(args.exp,
                                                           activation)
    net = bnn(
        X_train,
        y_train,
        ([int(n_hidden)] * num_hidden_layers),
        normalize=normalize,
        tau=tau,
        dropout=dropout,
        activation='relu'
    )

    print("Training model with {}...".format(activation))
    net.train(X_train, y_train, epochs=epochs, batch_size=batch_size, 
              verbose=1)

    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    net.model.save("{}model.h5".format(experiment_dir))

    plot_predictions(net, trainset, X_test, iters=20, n_std=4)
    plt.savefig("{}plot.png".format(experiment_dir))
