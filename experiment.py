

class Experiment:

    def __init__(self, 
                 activation='relu',
                 epochs=100):
        
        # model definition
        self.X_train = X_train
        self.y_train = y_train
        self.n_hidden = n_hidden
        self.num_hidden_layers = num_hidden_layers
        self.activation = activation
        self.dropout = dropout
        self.tau = tau
        self.normalize = normalize
        self.model = bnn(
                X_train,
                y_train,
                ([int(n_hidden)] * num_hidden_layers),
                normalize=False,
                tau=tau,
                dropout=dropout,
                activation=activation
        )

        # training
        self.epochs = epochs
        self.batch_size = batch_size


def run_experiment(activation):
    pass