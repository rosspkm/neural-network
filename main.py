# neural network start
import numpy as np
import pandas as pd



class data:

    def __init__(self, data:dict, labels:list):
        self.data = data # dict by index as key with row as list in value as numpy array
        self.labels = labels # list of labels for test data for training
    
    def get_data(self):
        return self.data # returns dict of numpy arrays
    
    def get_labels(self):
        return self.labels # returns list of labels


class NeuralNetwork:

    def __init__(self, input_size:int):
        self.weights1 = np.random.rand(16,input_size) # calculate random weights for layer 1
        self.weights2 = np.random.rand(8,16) # calculate random weights for layer 2
        self.weight_output = np.random.rand(10,8) # calculate random weights for output layer

    def calc_all_layers(self, img:list):

        # def __sigmoid(data:list): # element wise sigmoid function
        #     return (1/(1+(np.exp(-data))))

        def __relu(data:list):
            return np.maximum(0, data) 

        def __calc_layer1(): # calculate layer 1
            return __relu(np.dot(self.weights1, img))
        
        def __calc_layer2(): #c calculate layer 2
            return __relu(np.dot(self.weights2, __calc_layer1()))
        
        def __calc_output(): # calculate final output
            return __relu(np.dot(self.weight_output, __calc_layer2()))

        return __calc_output() # returns array of floats
    
    def make_target(self, target_cat:int):
        target_arr = np.zeros(10)
        target_arr[target_cat] = 1
        return target_arr

    def calculate_cost(self, target_cat:int, img:list):
        act_output = self.calc_all_layers(img)
        target_output = self.make_target(target_cat) # [0,0,1,0,0,0,0,0,0,0]
        cost = 0
        for i in range(len(act_output)):
            cost += ((act_output[i] - target_output[i])**2)/2
        return cost
    
    def output_layer_gradient(self, target_cat:int, img:list):
        y = self.make_target(target_cat=target_cat)
        z = self.calc_all_layers(img=img)
        return (y-z)
    




# make this a function eventually
file = "./data/train.csv"
df = pd.read_csv(file)
labels = df['label'].to_numpy()
df.drop(['label'], axis=1, inplace=True)

MNIST = data(data={i: df.iloc[i].to_numpy() for i in range(0, len(df.index))}, labels=labels)

NN = NeuralNetwork(len(MNIST.get_data()[0])+1)
#for i in range(0, len(MNIST.get_data())):

testImg = np.concatenate([[1], MNIST.get_data()[1]])
print(NN.output_layer_gradient(MNIST.get_labels()[0], testImg))
# labels[1]