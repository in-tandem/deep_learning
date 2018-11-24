import numpy as np 


class BackPropagationNeuralNetwork(object):

    def __init__(self, learning_rate, epochs, batchsize = 50, hidden_layers = 2):
        """

            TODO
        """

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batchsize = batchsize
        self.hidden_layers = hidden_layers
        self.biases = None
        self.weights = None

        ## initialize biases randomly. say network is defined as [4,3,1]. 
        ## Then bias matrix would be (3 * 1) for the first hidden layer
        ## bias matrix would (1*1) for the final output layer
        # self.biases = [ np.random.randn(i, 1) for i in sizes[1:] ] ## (i, 1) initializes bias as matrix of col 1


        ## initialize the weights matrix. say network is defined as [ 4, 3 , 1]
        ## weights matrix would be (4*3) for the first hidden layer
        ## weights matrix would be (3*1) for the output layer
        # self.weights = [ np.random.randn(y, x) for x, y in zip(sizes[: -1 ], sizes[1:]) ]



    def feedforward(self):
        pass
    
    def backpropagate(self):
        pass

    
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


    def initializeWeights(self,number_of_neurons, number_of_layers, number_of_output_neurons):
        """
            this is working off the assumptions that the number of neurons in input layer
            and number of neurons in the hidden layers are the same

            say number of neurons in each layer is 3, weights matrix should be 3 *3 for each
            layer.

            Now, in case the number of neurons are diff in each layer, then the weight matric would
            be (no of neurons in previous) * (no of neurons in the latter)
            
        """
        self.weights = []

        ## adding random weights for the hidden layers
        for i in range(number_of_layers):

            self.weights.append(np.random.randn(number_of_neurons, number_of_neurons))

        ## adding random weights for the output layer
        self.weights.append(np.random.randn(number_of_output_neurons, 1))


    def initializeBiases(self,number_of_neurons, number_of_layers, number_of_output_neurons):
        """
            this is working off the assumptions that the number of neurons in input layer
            and number of neurons in the hidden layers are the same

            say number of neurons in each layer is 3. then bias matrix should be 3 *1 for each
            hidden layer

        """
        self.biases = []

        ## adding random biases for the hidden layers
        for i in range(number_of_layers):

            self.biases.append(np.random.randn(number_of_neurons, 1))

        ## adding random biases for the output layer
        self.biases.append(np.random.randn(number_of_output_neurons, 1))



    def fit(self, x, y):

        for epoch in self.epochs:
            
            ## for each layer, calc the activation function and feed forward
            ## when at the end layer or output layer, calculate the error and back propagate
            ## while back propagating calculate the derivatives and update weights and bias accordingly

            pass


    def predict(self, x):
        pass







