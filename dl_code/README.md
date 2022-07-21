# Deep Learning for Uniformity Inversion

Several notebooks will walk through the deep learning inversion process.

- The first notebook, ``maxWaveNet_local.ipynb`` explains step by step how the given deep learning
model was created. 
- ``validate_network`` allows to analyze both the dataset and the trained deep net. It is 
possible to either push single dispersion images through the network to receive the single
output or to use validation batches and obtain performance metrics from that
- ``test_cuda_on_cluster.ipynb`` is a helpful tool to check whether GPU support is available
or not
