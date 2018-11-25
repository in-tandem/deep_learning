import numpy as np 

##TODO : mini batches and cost function at end of each epoch
class BackPropagationNeuralNetwork(object):

    def __init__(self, learning_rate, epochs, batchsize = 50, hidden_layers = [30]):
        """

        :param hidden_layers : this describes the number of hidden layers and the neurons in each
                            eg: [2,4,1] means there are 3 hidden layers, first hidden layer has 2 neurons
                                , second hidden layer has 4 and last hidden layer has 1 neuron

                                default parameter is 1 hidden layer with 30 neurons

        :type hidden_layers : list


        """

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batchsize = batchsize
        self.hidden_layers = hidden_layers
        self.biases = None
        self.weights = None


    def feedforward(self, x, y):
        
        self.activations_at_each_layer = []
        self.activation_derivative_at_each_layer = []
        
        neurons_at_each_layer = x
        
        for layer in range(len(self.hidden_layers + 1)):

            z = np.dot(self.weights[layer].T, neurons_at_each_layer) + self.biases[layer]

            self.activations_at_each_layer.append(self.calculateActivation(z))
            self.activation_derivative_at_each_layer.append(self.calculateDerivativeOfActivation(z))

            neurons_at_each_layer = z

        return (self.calculateErrorDerivativeAtOutputLayer(neurons_at_each_layer, y), self.calculateDerivativeOfActivation(z))



    def calculateErrorDerivativeAtOutputLayer(self, z, y):

        return np.dot(z -y,  self.calculateDerivativeOfActivation(z))

        
    
    def backpropagate(self, error_cost):
        
        error_at_current_layer = error_cost[0]

        error_cost_at_current_layer = error_cost[1]

        for layer in range(len(self.hidden_layers)):

            activation_at_previous_layer = self.activations_at_each_layer[: 0 - layer]


            error_at_previous_layer = np.dot(error_at_current_layer , self.weights[: 0 - layer]) * self.activation_derivative_at_each_layer[: 0 - layer]


            self.weights[: 0 - layer ] = self.weights[: 0 - layer ] - (self.learning_rate * np.dot(error_at_current_layer, activation_at_previous_layer))

            self.biases[: 0 - layer] = self.biases[: 0 - layer] - (self.learning_rate * error_at_current_layer)

            error_at_current_layer = error_at_previous_layer




    def initializeWeights(self,number_of_neurons, number_of_layers, number_of_output_neurons):
        """
           
            say number of neurons in each layer is 3, weights matrix should be 3 *3 for each
            layer.

            Now, in case the number of neurons are diff in each layer, then the weight matric would
            be (no of neurons in previous) * (no of neurons in the latter)

        """
        self.weights = []

        ## setting up number of layers to have the value for number of input layer neurons. 
        ## this will simplify our zip function
        number_of_layers = number_of_neurons + number_of_layers

        ## adding random weights for the hidden layers
        self.weights.extend([np.random.randn(y, x) for x,y in zip(number_of_layers[: -1], number_of_layers[: 1])])

        ## adding random weights for the output layer
        self.weights.append(np.random.randn(number_of_neurons,number_of_output_neurons))


    def initializeBiases(self, number_of_layers, number_of_output_neurons):
        """

            say number of neurons in each layer is 3. then bias matrix should be 3 *1 for each
            hidden layer

        """
        self.biases = []

        ## adding random biases for the hidden layers
        self.biases.extend([np.random.randn(i, 1) for i in number_of_layers])

        ## adding random biases for the output layer
        self.biases.append(np.random.randn(number_of_output_neurons, 1))


    
    def calculateActivation(self, z):
        """
        
        calculating the activation unit of z

        """

        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def calculateDerivativeOfActivation(self, z):
        """
        
            We are going to be using the logistic regression
            activation function where theta(z) = 1 / (1 + exp(-z))
            the partial derivative of the activation function
            wrt z is theta(z) * (1 - theta(z))

        """
        return self.calculateActivation(z) * ( 1 - self.calculateActivation(z))


    def fit(self, x, y):
        """

        :param x: this is a n * m matrix. n is the number of samples, m is the number of features
        """
        
        self.initializeBiases(self.hidden_layers, y.shape[0])
        self.initializeWeights(x.shape[1], self.hidden_layers, y.shape[0])

        for epoch in self.epochs:
            
            ## for each layer, calc the activation function and feed forward
            ## when at the end layer or output layer, calculate the error and back propagate
            ## while back propagating calculate the derivatives and update weights and bias accordingly
            ## shuffle at the start

            error_cost_at_output_layer = self.feedforward(x, y)

            self.backpropagate(error_cost_at_output_layer)



    def predict(self, x):
        pass







