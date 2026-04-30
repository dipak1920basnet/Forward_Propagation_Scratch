import numpy as np 
def linear(w,x,b):
    return np.dot(w,x) + b 

def sigmoid(z):
    return 1/(1+np.exp(-z))

def generate_w_b(row,column):
    weight = np.random.rand(row, column)
    bias = np.random.rand(1, column)
    return weight, bias

def dense(n,x):
    w,b = generate_w_b(x.shape[1],n)
    w = w.T
    
    matrix_multiplication = np.matmul(x,w)
    linear_value = matrix_multiplication + b
    return (sigmoid(linear_value).T)    

data = np.array(
    [[1,2,3],
    [4,5,6]]
)

print(dense(3, data))