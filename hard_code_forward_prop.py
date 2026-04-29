from model import activation
import numpy as np 

X = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

l1_a1 = activation(X)
l1_a2 = activation(X)
l1_a3 = activation(X)

layer_2_input = np.array([l1_a1, l1_a2, l1_a3]).T
for i in layer_2_input:
    print(i)
l2_a1 = activation(layer_2_input)
l2_a2 = activation(layer_2_input)

layer_3_input = np.array([l2_a1, l2_a2]).T
print()
print()
for i in layer_3_input:
    print(i)
output = activation(layer_3_input)

print("final output:: ",output)