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
        
        self.activations_at_each_layer = [x]
        self.activation_derivative_at_each_layer = [x]
        
        neurons_at_each_layer = x
        
        for layer in range(len(self.hidden_layers) + 1):

            z = np.dot(neurons_at_each_layer , self.weights[layer]) + self.biases[layer]

            self.activations_at_each_layer.append(self.calculateActivation(z))
            self.activation_derivative_at_each_layer.append(self.calculateDerivativeOfActivation(z))

            neurons_at_each_layer = z

        return (self.calculateErrorDerivativeAtOutputLayer(neurons_at_each_layer, y), self.calculateDerivativeOfActivation(z))



    def calculateErrorDerivativeAtOutputLayer(self, z, y):

        return (z -y) * self.calculateDerivativeOfActivation(z)

        
    
    def backpropagate(self, error_cost):
        
        error = error_cost[0]

        for i in range( len(self.hidden_layers)+1)[:: -1]:

            old_weight = self.weights[i]

            self.weights[i] = self.weights[i] - (self.learning_rate * np.dot(self.activations_at_each_layer[i].T, error))
            self.biases[ i] = self.biases[ i ] - (self.learning_rate * error)
            error = np.multiply(np.dot(error, old_weight.T), self.activation_derivative_at_each_layer[i])



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
        a = [number_of_neurons]
        a.extend(number_of_layers)
        a.append(number_of_output_neurons)

        ## adding random weights for the hidden layers
        self.weights.extend([np.random.randn(y, x) for x,y in zip(a[1:], a[: -1])])

        print(self.weights)
        ## adding random weights for the output layer
        # self.weights.append(np.random.randn(number_of_neurons,number_of_output_neurons))


    def initializeBiases(self, number_of_samples, number_of_layers, number_of_output_neurons):
        """

            say number of neurons in each layer is 3. then bias matrix should be 3 *1 for each
            hidden layer

        """
        self.biases = []

        # self.biases.append(np.random.randn(number_of_samples, 1))

        ## adding random biases for the hidden layers
        self.biases.extend([np.random.randn(i) for i in number_of_layers])

        ## adding random biases for the output layer
        self.biases.append(np.random.randn(number_of_output_neurons))


    
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
        self.cost_matrix = {}
        number_of_output_neurons = np.unique(y).shape[0]
        self.initializeBiases(x.shape[0], self.hidden_layers, number_of_output_neurons)
        self.initializeWeights(x.shape[1], self.hidden_layers, number_of_output_neurons)

        y = self.one_hot_encode_target(y)

        for epoch in range(self.epochs):
            
            ## for each layer, calc the activation function and feed forward
            ## when at the end layer or output layer, calculate the error and back propagate
            ## while back propagating calculate the derivatives and update weights and bias accordingly
            ## shuffle at the start

            error_cost_at_output_layer = self.feedforward(x, y)

            predictions_at_this_epoch = self.activations_at_each_layer[len(self.activations_at_each_layer) - 1]

            # predicted_y = np.argmax(predictions_at_this_epoch, axis =1)

            self.cost_matrix[epoch] = self.calculateCost(y, predictions_at_this_epoch)

            self.backpropagate(error_cost_at_output_layer)



    def predict(self, x):
        pass


    def calculateCost(self, y, y_predicted):

        term1 = - y * np.log(y_predicted)
        term2 = ( 1 - y) * np.log(1 - y_predicted)
        return np.sum(term1 -  term2)


    def one_hot_encode_target(self, y):

        onehot = np.zeros((np.unique(y).shape[0], y.shape[0]))

        for i , val in enumerate(y.astype(int)):

            onehot[val, i] = 1

        return onehot.T






