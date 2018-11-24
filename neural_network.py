import numpy as np 


class BackPropagationNeuralNetwork(object):

    def __init__(self, learning_rate, epochs, sizes, batchsize = 50):
        """

        :param sizes :  depicts the no. of neurons in each layer. 
                        so [4,3,1] depicts 3 layer network with 4 in 
                        input layer, 3 neurons in hidden layer and 1 in output layer
        
        :type sizes: list

        """

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.number_of_layers = len(sizes) ## no of elements in this list is no of layers
        self.batchsize = batchsize

        ## initialize biases randomly. say network is defined as [4,3,1]. 
        ## Then bias matrix would be (3 * 1) for the first hidden layer
        ## bias matrix would (1*1) for the final output layer
        self.biases = [ np.random.randn(i, 1) for i in sizes[1:] ] ## (i, 1) initializes bias as matrix of col 1


        ## initialize the weights matrix. say network is defined as [ 4, 3 , 1]
        ## weights matrix would be (4*3) for the first hidden layer
        ## weights matrix would be (3*1) for the output layer
        self.weights = [ np.random.randn(y, x) for x, y in zip(sizes[: -1 ], sizes[1:]) ]



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

        
    def fit(self, x, y):

        for epoch in self.epochs:
            
            for batch in self.batchsize:

                pass

                
            

    def predict(self, x):
        pass







