from neural_network import BackPropagationNeuralNetwork
from load_mnist import load_from_pickled_form, pickled_form
import numpy as np 
from matplotlib import pyplot as plot 

data = load_from_pickled_form(path = pickled_form)

x_train = data['x_train']
x_test = data['x_test']
y_train = data['y_train']
y_test = data['y_test']

back_propagation = BackPropagationNeuralNetwork(learning_rate = 0.01, epochs = 10)
back_propagation.fit(x_train[:100], y_train[:100])
