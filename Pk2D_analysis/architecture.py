import torch 
import torch.nn as nn
import numpy as np
import sys, os, time
import optuna

# This routine returns an architecture that is built inside the routine itself
# It can have from 1 to max_layers hidden layers. The user specifies the size of the
# input and output together with the maximum number of neurons in each layers
# trial -------------> optuna variable
# input_size --------> size of the input
# output_size -------> size of the output
# max_layers --------> maximum number of hidden layers to consider (default=3)
# max_neurons_layer -> the maximum number of neurons a layer can have (default=500)
def dynamic_model(trial, input_size, output_size, max_layers=3, max_neurons_layers=500):

    # define the tuple containing the different layers
    layers = []

    # get the number of hidden layers
    n_layers = trial.suggest_int("n_layers", 1, max_layers)

    # get the hidden layers
    in_features = input_size
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, max_neurons_layers)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.LeakyReLU(0.2))
        p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.8)
        layers.append(nn.Dropout(p))
        in_features = out_features

    # get the last layer
    layers.append(nn.Linear(out_features, output_size))

    # return the model
    return nn.Sequential(*layers)
