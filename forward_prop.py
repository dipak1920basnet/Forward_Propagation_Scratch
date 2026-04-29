from neural_network import NN
import numpy as np

X = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

neural_network = NN()
layer_and_neuron = [3,2,1]
for i in layer_and_neuron:
    neural_network.add_sequence(i)
neural_network.fit(X)
# neural_network.final_output_neuron()

activation_list = neural_network.activation_list
# print(activation_list)
for i in range(len(activation_list)):
    layer_ = activation_list[i]
    
    if i == 0:
        print("data layer: ", layer_)
        continue

    print(f"activation{i}:: ",layer_)

    # for j in range(len(layer_)):
    # # print(f"layer{i}:: {activation_list[i].T}")
    #     print(f"layer{i}, perceptron{j} activation: {layer_[j]}")