
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
from viz import plot_predictions_no_legend

parser = argparse.ArgumentParser(description='Mauna Loa experiment runner')
parser.add_argument('-e', '--exp', default=0, type=int, help='Experiment number')
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

num_hidden_layers = 4
n_hidden = 1024  # num hidden units
epochs = 50
batch_size = 128
tau = 10
lengthscale=5
optimizer='adam'
dropout = 0.1
normalize = False
test_iters = 1000

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

fig, axes = plt.subplots(len(activations), sharex=True, sharey=True)
fig.set_figwidth(7.48)
fig.set_figheight(1.75 * len(activations))
fig.tight_layout()
fig.subplots_adjust(hspace=0)
plt.xlim(-1.75, 3.75)

# legend
custom_lines = [Line2D([0], [0], color="r", lw=4),
                Line2D([0], [0], color="k", lw=1),
                Line2D([0], [0], color="b", lw=4)]
fig.legend(custom_lines, ["observed", "mean", "uncertainty"], ncol=3,
           bbox_to_anchor=(0.075, .01, .89, .01), loc="lower left",
           mode="expand", borderaxespad=.02, fancybox=True, shadow=True
           )

experiment_dir = "experiments/mauna_loa/"

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
        activation=activation
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

    # create prediction plot
    ax = axes[a]
    plot_predictions_no_legend(net, trainset, X_test, iters=test_iters,
                               n_std=2, ax=ax)
    ax.set(ylabel=activation)
    ax.label_outer()

# save stats
df = pd.DataFrame.from_dict(stats, orient='index')
df.to_csv("{}stats.csv".format(experiment_dir))

# adjust bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
# plt.subplots_adjust(bottom=0.1)

# save plots
figname = "{}plot".format(experiment_dir)
plt.savefig("{}.png".format(figname))
# save reloadable/editable plot
with open("{}.fig.pickle".format(figname), "wb") as f:
    pickle.dump(fig, f)
