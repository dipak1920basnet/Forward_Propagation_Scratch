import numpy as np 
def linear(w,x,b):
    return np.dot(w,x) + b 

def sigmoid(z):
    return 1/(1+np.exp(-z))

def generate_w_b(row,column):
    weight = np.random.rand(row, column)
    bias = np.random.rand(1, column)
    return weight, bias

  

data = np.array(
    [[1,2,3],
    [4,5,6]]
)

class NN:
    def __init__(self):
        self.sequence = []
        self.activation_list = []
        self.weight_list = []
        self.bias_list = []
        self.predict_store = []


    def Dense(self,x,w,b):
        """
        n: the total number of neuron or perceptron
        x: input data
        """
        matrix_multiplication = np.matmul(x,w)
        z = matrix_multiplication + b
        a = sigmoid(z)
        return (a)  

    def Sequence(self,dense_list:list):
        self.sequence = dense_list

    def fit(self, X):
        """
        Currently this only does the forward pass no learning. 
        """

        # set the input data to activation 0
        self.activation_list = []
        self.weight_list = []
        self.bias_list = []

        self.activation_list.append(X)

        for i in self.sequence:
            data = self.activation_list[-1]
            w,b = generate_w_b(data.shape[1],i)
            print(f"activation_:: ",data)
            print("weight:: ", w)
            print("bias:: ", b)
            print()
            print()
            new_activation = self.Dense(x=data, w=w,b=b)

            self.activation_list.append(new_activation)
            self.weight_list.append(w)
            self.bias_list.append(b)

    def predict(self, X):
        self.predict_store = []
        self.predict_store.append(X)
        for i in range(len(self.weight_list)):
            data = self.predict_store[-1]
            prediction = np.matmul(data, self.weight_list[i]) + self.bias_list[i]
            self.predict_store.append(sigmoid(prediction))
        return self.predict_store[-1]

model = NN()
model.Sequence([
    3,2,1
])
model.fit(data)
prediction = model.predict(data)
print(prediction)
print("weight list ",model.weight_list)
print("bias list ",model.bias_list)