# Restriced Boltzmann Machines with Tensorflow

The model is Boltzmann Machine-like: it has three layers, each without internal connections, and the energy functional is invariant under permutation of two of the layers. 

The model resembles a Restricted Boltzmann Machine with two visible layers having the same shape and type, connected to each other by a symmetric interaction matrix, and sharing the same biases and interaction matrix with the hidden layer.

This allows to train the model to recognize associations between pairs of data points. For instance, it can be used to learn protein-protein interactions.

A simpler example (counting_test.py) is included: data points are binary vectors, and the implicit association rule is that for two vectors to be associated both must have the same number of "on" units. If given a new pair of vectors, the model can accurately predict a high/low free energy, then it can be said that its internal representation is in effect learning to count!
