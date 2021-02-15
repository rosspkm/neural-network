# neural network start
import numpy as np
import pandas as pd
from lib import classes

class NeuralNetwork:

    def __init__(self, input_size:int):
        self.weights1 = np.random.rand(16,input_size) # calculate random weights for layer 1
        self.weights2 = np.random.rand(8,16) # calculate random weights for layer 2
        self.weight_output = np.random.rand(10,8) # calculate random weights for output layer

        self.layer1_output = None
        self.layer2_output = None
        self.output_layer_output = None

    def calc_all_layers(self, img:list):
        # def __sigmoid(data:list): # element wise sigmoid function
        #     return (1/(1+(np.exp(-data))))

        def __relu(data:list):
            return np.maximum(0, data) 

        def __calc_layer1(): # calculate layer 1
            self.layer1_output = __relu(data=np.dot(self.weights1, img))
            return self.layer1_output
        
        def __calc_layer2(): # calculate layer 2
            self.layer2_output = __relu(data=np.dot(self.weights2, __calc_layer1()))
            return self.layer2_output
        
        def __calc_output(): # calculate final output
            self.output_layer_output =  __relu(data=np.dot(self.weight_output, __calc_layer2()))
            return self.output_layer_output

        return __calc_output() # returns array of floats
    
    def make_target(self, target_cat:int):
        target_arr = np.zeros(10)
        target_arr[target_cat] = 1
        return target_arr

    def calculate_cost(self, target_cat:int, img:list):
        act_output = self.calc_all_layers(img=img)
        target_output = self.make_target(target_cat=target_cat) # [0,0,1,0,0,0,0,0,0,0]
        cost = 0
        for i in range(len(act_output)):
            cost += ((act_output[i] - target_output[i])**2)/2
        return cost
    
    # returns the partial derivatives for each weight in the output layer with respect to the cost function
    # the cost function is 1/2 * (actual output - target output)**2
    # the derivative of that is (target output - actual output) 
    def output_layer_gradient(self, target_cat:int, img:list):
        y = self.make_target(target_cat=target_cat)
        z = self.calc_all_layers(img=img)
        return (y-z)

    def layer2_gradient(self, target_cat:int, img:list):
        temp = self.output_layer_gradient(target_cat=target_cat, img=img)
        temp2 = self.layer2_output
        weights = self.weights2
        

        # [w1, w2, w3] = [0]
        # [w4, w5, w6] = [20]
        # [w7, w8, w9] = [15]

        # need output of layer2, weights2





# make this a function eventually
file = "./data/train.csv"
df = pd.read_csv(file)
labels = df['label'].to_numpy()
df.drop(['label'], axis=1, inplace=True)

MNIST = classes.data(data={i: df.iloc[i].to_numpy() for i in range(0, len(df.index))}, labels=labels)

NN = NeuralNetwork(len(MNIST.get_data()[0])+1)
#for i in range(0, len(MNIST.get_data())):

testImg = np.concatenate([[1], MNIST.get_data()[1]])
print(NN.output_layer_gradient(MNIST.get_labels()[0], testImg))
# labels[1]