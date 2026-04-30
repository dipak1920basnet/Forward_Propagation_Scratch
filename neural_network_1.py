import numpy as np

np.random.seed(42)

def sigmoid(z):
    # get a sigmoid activations
    return 1 / (1 + np.exp(-z))


def generate_w_b(row, column):
    """
    generate weight and bias that matches row and column based on data and dense of layers.
    no need to transpose the weight and bias as this already  matches row and column for data. 
    """
    weight = np.random.randn(row, column)
    bias = np.random.randn(1, column)
    return weight, bias


class NN:
    def __init__(self):
        self.sequence = []
        self.activation_list = []
        self.weight_list = []
        self.bias_list = []
        self.predict_store = []

    def Dense(self, x, w, b):
        """
        n: the total number of neuron or perceptron
        x: input data
        """
        matrix_multiplication = np.matmul(x, w)
        z = matrix_multiplication + b
        a = sigmoid(z)
        return a

    def Sequence(self, dense_list: list):
        """
        set the list of layers and dense inside layers
        """
        self.sequence = dense_list

    def fit(self, X):
        """
        Currently this only does the forward pass no learning.
        """

        # avoid double stack if fit is called multiple times.
        self.activation_list = []
        self.weight_list = []
        self.bias_list = []

        # set the input data to activation 0
        self.activation_list.append(X)

        for i in self.sequence:

            # Call the latest activations
            data = self.activation_list[-1]
            w, b = generate_w_b(data.shape[1], i)
            print(f"activation_:: ", data)
            print("weight:: ", w)
            print("bias:: ", b)
            print()
            print()

            # get new activation for layers
            new_activation = self.Dense(x=data, w=w, b=b)

            # store the activation for forward prop
            self.activation_list.append(new_activation)

            # store the weight and bias for prediction of data
            self.weight_list.append(w)
            self.bias_list.append(b)

    def predict(self, X):
        self.predict_store = []
        # set the data as activation 0 as input layer is activation 0
        self.predict_store.append(X)
        for i in range(len(self.weight_list)):
            # get the latest activation
            data = self.predict_store[-1]

            # access the weight and bias for particular layer with i and perform predictions. 
            prediction = self.Dense(data, self.weight_list[i], self.bias_list[i])

            # store the acctivation 
            self.predict_store.append(prediction)
        
        # return the predictions
        return self.predict_store[-1]


data = np.array([[1, 2, 3], [4, 5, 6]])


# instiate the model
model = NN()
# Set the layers
model.Sequence([3, 2, 1])
# Fit the data
model.fit(data)
# predict the data
prediction = model.predict(data)

print("predictions:: ",prediction)
print("weight list ", model.weight_list)
print("bias list ", model.bias_list)
