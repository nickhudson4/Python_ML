import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math

class NeuralNetwork:
    def __init__(self, numLayers, layers, learning_rate):
        self.numLayers = numLayers
        self.layers = layers
        self.learning_rate = learning_rate

    def get_cost(self, values, correct_num):
        sum = 0
        for i in range(len(values)):
            if i == correct_num:
                sum += (values[i] - 1.0)**2
            else:
                sum += (values[i] - 0.0)**2

        return sum


class Layer:
    def __init__(self, size, networkPos, neurons, weights, biases, values, isInputLayer):
        self.size = size
        self.networkPos = networkPos
        self.neurons = neurons
        self.weights = weights
        self.biases = biases
        self.values = values
        self.isInputLayer = isInputLayer

class Neuron:
    def __init__(self, connectedNeurons, ID): # "weights" is a list linked with "connectedNeurons"
        self.connectedNeurons = connectedNeurons
        self.ID = ID

def setup_network(x_train, x_test, y_train, y_test, numPixels_x, numPixels_y):
    totalPixels = numPixels_x * numPixels_y
    hiddenLayerSize = 16

    layer_input_neurons = []
    layer_2_neurons = []
    layer_3_neurons = []
    layer_output_neurons = []
    neuron_id = 0
    for i in range(numPixels_x): # Create input layer
        for j in range(numPixels_y):
            neuron = Neuron([], neuron_id)
            layer_input_neurons.append(neuron)
            neuron_id += 1

    for i in range(16): # Create second "hidden" layer
        neuron = Neuron([], neuron_id)
        layer_2_neurons.append(neuron)
        neuron_id += 1

    for i in range(16): # Create third "hidden" layer
        neuron = Neuron([], neuron_id)
        layer_3_neurons.append(neuron)
        neuron_id += 1

    for i in range(10): # Create output layer
        neuron = Neuron([], neuron_id)
        layer_output_neurons.append(neuron)
        neuron_id += 1

    for i in range(len(layer_2_neurons)):
        for j in range(len(layer_input_neurons)):
            layer_2_neurons[i].connectedNeurons.append(layer_input_neurons[j])

    for i in range(len(layer_3_neurons)):
        for j in range(len(layer_2_neurons)):
            layer_3_neurons[i].connectedNeurons.append(layer_2_neurons[j])

    for i in range(len(layer_output_neurons)):
        for j in range(len(layer_3_neurons)):
            layer_output_neurons[i].connectedNeurons.append(layer_3_neurons[j])

    layer_0 = Layer(totalPixels, 0, layer_input_neurons, [], [], [], True)
    layer_1 = Layer(hiddenLayerSize, 1, layer_2_neurons, [], [], [], False)
    layer_2 = Layer(hiddenLayerSize, 2, layer_3_neurons, [], [], [], False)
    layer_3 = Layer(10, 3, layer_output_neurons, [], [], [], False)

    layer_set = [layer_0, layer_1, layer_2, layer_3]
    neural_network = NeuralNetwork(4, layer_set, 0.1)

    return neural_network

def setup_first_generation(network, new_image, numPixels_x, numPixels_y):
    for i in range(len(network.layers)):
        values = []
        if i == 0: # Setup input layer
            values = np.zeros((numPixels_x * numPixels_y, 1))
            neuron_index = 0
            for n in range(numPixels_x):
                for m in range(numPixels_y):
                    values[neuron_index] = new_image[n][m]
                    neuron_index += 1

        else: # All other layers
            biases = np.zeros((len(network.layers[i].neurons) , 1))
            values = np.zeros((len(network.layers[i].neurons), 1))
            for n in range(len(network.layers[i].neurons)):
                values[n] = 0
                biases[n] = random.uniform(-1.0, 1.0)
                row_weights = []
                for m in range(len(network.layers[i].neurons[n].connectedNeurons)):
                    row_weights.append(random.uniform(-1.0, 1.0))
                network.layers[i].weights.append(row_weights)
            network.layers[i].weights = np.array(network.layers[i].weights)
            network.layers[i].biases = biases 

        network.layers[i].values = values

def setup_input_layer(network, new_image, numPixels_x, numPixels_y):
    values = np.zeros((numPixels_x * numPixels_y, 1))
    neuron_index = 0
    for n in range(numPixels_x):
        for m in range(numPixels_y):
            values[neuron_index] = new_image[n][m]
            neuron_index += 1
    
    network.layers[0].values = values


# * Change this to use vectors
def get_hidden_layer_activation(weights, connectedNeurons, bias):
    if len(weights) != len(connectedNeurons):
        print("Error Calculating Activation for Hidden Layer")
    
    sum = 0
    for i in range(len(weights)):
        sum += weights[i] * connectedNeurons[i].value
    sum -= bias

    return sigmoid(sum) #Normalize result

def sigmoid(x):
    sigm = 1. / (1. + np.exp(-x))
    return sigm

def sigmoidPrime(s):
    #derivative of sigmoid
    return s * (1 - s)

def draw_network(network):
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    neuronSize = (1/96)
    for i in range(len(network.layers[1].neurons)):
        circle = plt.Circle((.5, i/18), neuronSize, fill=False)
        plt.gca().add_patch(circle)

    plt.show()
    print(network.layers[3].neurons[1].value)
    print(len(network.layers[1].neurons))

def activate(weights, values, biases):
    # print("WEIGHTS: ", len(weights[0]))
    # print("VALUES: ", len(values))
    
    multiply = np.dot(weights, values)
    # print("Multiply: ", multiply)
    # print("BIASES: ", biases)

    result = sigmoid(multiply + biases)
    # print("Result: ", result)
    return result

def forward_propagate(network):
    for i in range(len(network.layers)):
        if network.layers[i].isInputLayer:
            continue
        act_vals = activate(network.layers[i].weights, network.layers[i-1].values, network.layers[i].biases)
        network.layers[i].values = np.array(act_vals)
    
    return network.layers[3].values

def get_output_errors(outputs, target_num):
    target_vector = np.zeros((len(outputs), 1))
    target_vector[target_num] = 1.0
    return target_vector - outputs

def get_prediction(x):
    current_max = -1
    current_index = -1
    for i in range(len(x)):
        if x[i] > current_max:
            current_index = i
            current_max = x[i]
    
    return current_index


def main():
    mnist = tf.keras.datasets.mnist

    # * x_train is [image][row][column] 

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    numPixels_x = len(x_train[0])
    numPixels_y = len(x_train[0][0])

    neural_network = setup_network(x_train, x_test, y_train, y_test, numPixels_x, numPixels_y)
    setup_first_generation(neural_network, x_train[0], numPixels_x, numPixels_y)

    for p in range(1):
        for x in range(len(x_train)):
            setup_input_layer(neural_network, x_train[x], numPixels_x, numPixels_y)
            outputs = forward_propagate(neural_network)

            output_errors = get_output_errors(outputs, y_train[x])
            
            gradient = sigmoidPrime(outputs)
            gradient = np.multiply(output_errors, gradient)
            gradient = np.multiply(gradient, neural_network.learning_rate)

            hidden2_t = np.transpose(neural_network.layers[2].values)
            weights_ho_deltas = np.dot(gradient, hidden2_t)
            # Adjust output layer weights
            weights_ho = neural_network.layers[3].weights + weights_ho_deltas
            neural_network.layers[3].weights = weights_ho
            neural_network.layers[3].biases += gradient

            #Calculate the hidden layer errors
            who_t = np.transpose(weights_ho)
            hidden2_errors = np.dot(who_t, output_errors)
            #!

            #Calculate hidden gradient
            hidden2_gradient = sigmoidPrime(neural_network.layers[2].values)
            hidden2_gradient = np.multiply(hidden2_errors, hidden2_gradient)
            hidden2_gradient = np.multiply(hidden2_gradient, neural_network.learning_rate)

            #Calculate h --> h deltas
            hidden1_t = np.transpose(neural_network.layers[1].values)
            weights_hh_deltas = np.dot(hidden2_gradient, hidden1_t)

            weights_hh = neural_network.layers[2].weights + weights_hh_deltas
            neural_network.layers[2].weights = weights_hh
            neural_network.layers[2].biases += hidden2_gradient

            #Calculate the hidden layer errors
            whh_t = np.transpose(weights_hh)
            hidden1_errors = np.dot(whh_t, hidden2_errors)
            #!

            #Calculate hidden gradient
            hidden1_gradient = sigmoidPrime(neural_network.layers[1].values)
            hidden1_gradient = np.multiply(hidden1_errors, hidden1_gradient)
            hidden1_gradient = np.multiply(hidden1_gradient, neural_network.learning_rate)

            #Calculate h --> h deltas
            hidden0_t = np.transpose(neural_network.layers[0].values)
            weights_ih_deltas = np.dot(hidden2_gradient, hidden0_t)
            weights_ih = neural_network.layers[1].weights + weights_ih_deltas
            neural_network.layers[1].weights = weights_ih
            neural_network.layers[1].biases += hidden1_gradient

            #Calculate the hidden layer errors
            # wih_t = np.transpose(weights_ih)
            # hidden0_errors = np.dot(wih_t, hidden1_errors)
            #!



            print("OUTPUTS: ", outputs)
            print("Target: ", y_train[x])
            print("Prediction: ", get_prediction(outputs))
            print("Accuracy: ", neural_network.get_cost(outputs, y_train[x]))
            print("ERRORS: ", output_errors)
            print("GRADIENT: ", gradient)
            # print("HIDDEN_T: ", hidden2_t)
            # print("weights_delta: ", weights_ho_deltas)


        totalCost = 0
        for x in range(len(x_test)):
            setup_input_layer(neural_network, x_test[x], numPixels_x, numPixels_y)
            outputs = forward_propagate(neural_network)
            totalCost += neural_network.get_cost(outputs, y_test[x])


        print(totalCost)

    #* Debugging stuff below
    # draw_network(neural_network)

    # print("x_train: ", len(x_train[1][0]))
    # print("y_train: ", y_train[1])
    # plt.imshow(x_train[1], cmap = plt.cm.binary)
    # plt.show()

main()


        
