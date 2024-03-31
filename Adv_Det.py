## --Load Libraries-- ##
import argparse
import pathlib
from tensorflow.keras import layers
import sys
import numpy as np
from sklearn.decomposition import IncrementalPCA
from scipy.stats import wasserstein_distance,energy_distance
from sklearn.decomposition import PCA
import os
import matplotlib.pyplot as plt

### --Load Functions-- ##
from Prueba.functions.load_DS import load_ds
from Prueba.functions.load_model import load_model

### --LOAD DATASET-- ###
## Load the dataset to use in the neural network models. Posible datasets:
# 'mnist'             Available in all the models
# 'fashion_mnist'     Not available in Cauchy and Gumbal vgg16 models

## For the categories select a number between 0 and 9. Follow the dataset, the numbers rep´resent a especific category:
## MNIST Categories:
# 0   Class 0
# 1   Class 1
# 2   Class 2
# 3   Class 3
# 4   Class 4
# 5   Class 5
# 6   Class 6
# 7   Class 7
# 8   Class 8
# 9   Class 9
## FASHION-MNIST Categories:
# 0   Class T-shirt/top
# 1   Class Trouser
# 2   Class Pullover
# 3   Class Dress
# 4   Class Coat
# 5   Class Sandal
# 6   Class Shirt
# 7   Class Sneaker
# 8   Class Bag
# 9   Class Ankle boot
data = load_ds('mnist',5)

## This code allow the load of the model follow the kind of neural network. This is the list of possible models to load:
# 'vgg16_mnist'             Deterministic vgg16 model in mnist dataset
# 'vgg19_mnist'             Deterministic vgg19 model in mnist dataset
# 'vgg16bt_mnist'           Full MNF Bayesian vgg16 model in mnist dataset
# 'vgg19bt_mnist'           Full MNF Bayesian vgg19 model in mnist dataset
# 'vgg16b1_mnist'           Last layer MNF Bayesian vgg16 model in mnist dataset
# 'vgg19b1_mnist'           Last layer MNF Bayesian vgg19 model in mnist dataset
# 'vgg16bt_Re_mnist'        Full Reparameterization Trick Bayesian vgg16 model in mnist dataset
# 'vgg19bt_Re_mnist'        Full Reparameterization Trick Bayesian vgg19 model in mnist dataset
# 'vgg16b1_Re_mnist'        Last layer Reparameterization Trick Bayesian vgg16 model in mnist dataset
# 'vgg19b1_Re_mnist'        Last layer Reparameterization Trick Bayesian vgg19 model in mnist dataset
# 'vgg16_fashion'           Deterministic vgg16 model in fashion_mnist dataset
# 'vgg19_fashion'           Deterministic vgg19 model in fashion_mnist dataset
# 'vgg16bt_fashion'         Full MNF Bayesian vgg16 model in fashion_mnist dataset
# 'vgg19bt_fashion'         Full MNF Bayesian vgg19 model in fashion_mnist dataset
# 'vgg16b1_fashion'         Last layer MNF Bayesian vgg16 model in fashion_mnist dataset
# 'vgg19b1_fashion'         Last layer MNF Bayesian vgg19 model in fashion_mnist dataset
# 'vgg16bt_Re_fashion'      Full Reparameterization Trick Bayesian vgg16 model in fashion_mnist dataset
# 'vgg19bt_Re_fashion'      Full Reparameterization Trick Bayesian vgg19 model in fashion_mnist dataset
# 'vgg16b1_Re_fashion'      Last layer Reparameterization Trick Bayesian vgg16 model in fashion_mnist dataset
# 'vgg19b1_Re_fashion'      Last layer Reparameterization Trick Bayesian vgg19 model in fashion_mnist dataset
# 'vgg16bt_CA'              Full Cauchy MNF Bayesian vgg16 model in mnist dataset
# 'vgg16b1_CA'              Full Cauchy MNF Bayesian vgg16 model in mnist dataset
# 'vgg16bt_GUM'             Full Gumbel MNF Bayesian vgg16 model in mnist dataset
# 'vgg16b1_GUM'             Full Cumbel MNF Bayesian vgg16 model in mnist dataset
Model = load_model('vgg16_mnist')