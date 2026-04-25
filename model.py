import numpy as np 

def main():
    ...

def linear_model(w,x,b):
    return np.dot(x,w)+b

def sigmoid_model(z):
    denominator = 1 + (np.exp(-z))
    return 1 / denominator

def sigmoid(w,x,b):
    z = linear_model(w,x,b)
    return sigmoid_model(z)

def activation(x):
    m = x.shape[1]
    w = np.zeros(m)
    # initialize the base later. 
    b = np.random()

    return sigmoid(w,x,b)

if __name__ == "__main__":
    main()