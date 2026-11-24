# Code for Paper "Activity Pruning for Efficient Spiking Neural Networks"

## To run
First setup data-path, hyper-parameters in '.yaml' file.
For example,
eta: parameter for threshold regularization
eta2: parameter for activity regularization
test-only: whether to train or test

Train CIFAR-10 Models
> python main.py --config=cifar10.yaml


Train ImageNet-100 Models