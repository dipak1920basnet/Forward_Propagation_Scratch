from model import activation
import numpy as np
class NN:
    def __init__(self):
        self.sequence = []
        self.activation_list = []
    
    # the private class. and cannot be accessed from outside. 
    def perceptron(self, X):
        return activation(X)

    # add no of layers and neurons
    def add_sequence(self, no_of_neurons):
        self.sequence.append(no_of_neurons)
    
    def fit(self, X):
        # Activation 0 is the input layer so 
        self.activation_list.append(X)
        for layer in self.sequence:
            # access the latest activation
            input_ = self.activation_list[-1]

            # collect the output of each neuron
            new_activation_list = []

            for no_of_neuron in range(layer):
                prediction = self.perceptron(input_)
                new_activation_list.append(prediction)
            
            # append the latest activation
            self.activation_list.append(np.array(new_activation_list).T)

    # get the output of the nn. 
    def final_output_neuron(self):
        print("Activation of last layer:: ")
        print(self.activation_list[-1])