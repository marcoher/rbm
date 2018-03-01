# Restriced Boltzmann Machines with Tensorflow

We implement a Boltzmann Machine-like model with three layers, without internal connections, and where the energy functional is invariant under permutation of two of the layers.

The model resembles a Restricted Boltzmann Machine with two visible layers having the same shape and type, connected to each other by a symmetric interaction matrix, and sharing the same biases and interaction matrix with the hidden layer.

This allows to train the model to recognize associations between pairs of data points. For instance, it can be used to learn protein-protein interactions.

A simpler example (counting_test.py) is given where data points are binary vectors, and the implicit association rule is that for two vectors to be associated both must have the same number of "on" units. 
