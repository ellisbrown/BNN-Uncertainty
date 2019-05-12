import math
from scipy.misc import logsumexp
import numpy as np

import tqdm
import time
import os
import tensorflow as tf
import tensorflow_probability as tfp
from keras.regularizers import l2
from keras import Input
from keras.layers import Dropout
from keras.layers import Dense
from keras import Model


class bnn:

    def __init__(self, X_train, y_train, n_hidden,
                 normalize=False, tau=1.0, dropout=0.05,
                 lengthscale=1e-2, optimizer='adam',
                 weight_prior="glorot_uniform", bias_prior='zeros',
                 activation='relu'):
        """
            Constructor for the class implementing a Bayesian neural network
            trained with the probabilistic back propagation method.
            @param X_train      Matrix with the features for the training data.
            @param y_train      Vector with the target variables for the
                                training data.
            @param n_hidden     Vector with the number of neurons for each
                                hidden layer.
            @param n_epochs     Numer of epochs for which to train the
                                network. The recommended value 40 should be
                                enough.
            @param normalize    Whether to normalize the input features. This
                                is recommended unles the input vector is for
                                example formed by binary features (a
                                fingerprint). In that case we do not recommend
                                to normalize the features.
            @param tau          Tau value used for regularization
            @param dropout      Dropout rate for all the dropout layers in the
                                network.
        """

        # We normalize the training data to have zero mean and unit standard
        # deviation in the training set if necessary

        if normalize:
            self.std_X_train = np.std(X_train, 0)
            self.std_X_train[self.std_X_train == 0] = 1
            self.mean_X_train = np.mean(X_train, 0)
        else:
            self.std_X_train = np.ones(X_train.shape[1])
            self.mean_X_train = np.zeros(X_train.shape[1])

        X_train = (X_train - np.full(X_train.shape, self.mean_X_train)) / \
            np.full(X_train.shape, self.std_X_train)

        self.mean_y_train = np.mean(y_train)
        self.std_y_train = np.std(y_train)

        y_train_normalized = (y_train - self.mean_y_train) / self.std_y_train
        y_train_normalized = np.array(y_train_normalized, ndmin=2).T

        # We construct the network
        N = X_train.shape[0]
        batch_size = 128
        reg = lengthscale**2 * (1 - dropout) / (2. * N * tau)

        inputs = Input(shape=(X_train.shape[1],))
        inter = Dropout(dropout)(inputs, training=True)
        inter = Dense(n_hidden[0], activation=activation,
                      kernel_initializer=weight_prior,
                      bias_initializer=bias_prior,
                      kernel_regularizer=l2(reg))(inter)
        for i in range(len(n_hidden) - 1):
            inter = Dropout(dropout)(inter, training=True)
            inter = Dense(n_hidden[i+1], activation=activation,
                          kernel_initializer=weight_prior,
                          bias_initializer=bias_prior,
                          kernel_regularizer=l2(reg))(inter)
        inter = Dropout(dropout)(inter, training=True)
        outputs = Dense(
            y_train_normalized.shape[1],
            kernel_initializer=weight_prior,
            bias_initializer=bias_prior,
            kernel_regularizer=l2(reg))(inter)
        model = Model(inputs, outputs)

        self.tau = tau
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        self.model = model

    def train(self, X_train, y_train, batch_size, epochs=40, verbose=0):
        # normalize
        X_train = (X_train - np.full(X_train.shape, self.mean_X_train)) / \
            np.full(X_train.shape, self.std_X_train)
        y_train_normalized = (y_train - self.mean_y_train) / self.std_y_train
        y_train_normalized = np.array(y_train_normalized, ndmin=2).T

        # iterate training
        start_time = time.time()
        # self.model.fit(X_train, y_train,
        self.model.fit(X_train, y_train_normalized,
                       batch_size=batch_size, epochs=epochs, verbose=0)
        self.running_time = time.time() - start_time

    def predict(self, X_test, T=10000):
        X_test = np.array(X_test, ndmin=2)

        X_test = (X_test - np.full(X_test.shape, self.mean_X_train)) / \
            np.full(X_test.shape, self.std_X_train)

        pbar = tqdm.trange(T)

        Yt_hat = np.array(
            [self.model.predict(X_test, batch_size=500, verbose=0) for _ in pbar])
        Yt_hat = Yt_hat * self.std_y_train + self.mean_y_train

        # We compute the predictive mean and variance for the target variables
        # of the test data
        means = [Yt_hat[:, i].mean() for i in range(Yt_hat.shape[1])]
        stds = [Yt_hat[:, i].std() for i in range(Yt_hat.shape[1])]
        return np.stack(means), np.stack(stds)

    def test(self, X_test, y_test, T=10000):
        """
            Function for making predictions with the Bayesian neural network.
            @param X_test   The matrix of features for the test data
            @param T        The number of samples used for mean and var
                            prediction


            @return m       The predictive mean for the test target variables.
            @return v       The predictive variance for the test target
                            variables.
            @return v_noise The estimated variance for the additive noise.
        """

        X_test = np.array(X_test, ndmin=2)
        y_test = np.array(y_test, ndmin=2).T

        # We normalize the test set

        X_test = (X_test - np.full(X_test.shape, self.mean_X_train)) / \
            np.full(X_test.shape, self.std_X_train)

        # We compute the predictive mean and variance for the target variables
        # of the test data

        model = self.model
        standard_pred = model.predict(X_test, batch_size=500, verbose=1)
        standard_pred = standard_pred * self.std_y_train + self.mean_y_train
        rmse_standard_pred = np.mean(
            (y_test.squeeze() - standard_pred.squeeze())**2.)**0.5

        pbar = tqdm.tnrange(T)

        Yt_hat = np.array(
            [model.predict(X_test, batch_size=500, verbose=0) for _ in pbar])
        Yt_hat = Yt_hat * self.std_y_train + self.mean_y_train
        MC_pred = np.mean(Yt_hat, 0)
        rmse = np.mean((y_test.squeeze() - MC_pred.squeeze())**2.)**0.5

        # We compute the test log-likelihood
        ll = (logsumexp(-0.5 * self.tau * (y_test[None] - Yt_hat)**2., 0) -
              np.log(T) - 0.5*np.log(2*np.pi) + 0.5*np.log(self.tau))
        test_ll = np.mean(ll)

        # We are done!
        return rmse_standard_pred, rmse, test_ll
