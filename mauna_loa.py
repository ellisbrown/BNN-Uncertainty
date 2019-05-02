
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
from keras.models import load_model

from data import load_from_H5
from bnn import bnn
from viz import plot_predictions

parser = argparse.ArgumentParser(description='Mauna Loa runner')
parser.add_argument('--trained_model', default='models/mauna_loa.h5',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--train', default=False, type=bool)
parser.add_argument('--model_name', default='mauna_loa.h5', type=str,
                    help='Dir to save results')
parser.add_argument('--epochs', default=40, type=int, help='Number of epochs to train on')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()


test_hdf5_filepath = 'data/Mauna Loa/test.h5'
train_hdf5_filepath = 'data/Mauna Loa/train.h5'

testset = load_from_H5(test_hdf5_filepath)
trainset = load_from_H5(train_hdf5_filepath)

X_test, y_test = testset
X_train, y_train = trainset

N = 272
# l2 = 0.01
num_hidden_layers = 5
# num hidden units
n_hidden = 1024
epochs = args.epochs
batch_size = 128
epochs_multiplier = 1
epochs_multiplier
tau = 0.1
dropout = 0.1
net = bnn(
    X_train,
    y_train,
    ([int(n_hidden)] * num_hidden_layers),
    normalize=False,
    tau=tau,
    dropout=dropout,
    activation='relu'
)

if args.train:
    print("Training a new model...")
    net.train(X_train, y_train, epochs=epochs, batch_size=batch_size, 
              verbose=1)
    net.model.save("models/" + args.model_name)
else:
    net.model = load_model(args.trained_model)

plot_predictions(net, trainset, X_test, iters=20, n_std=4)
plt.show()
