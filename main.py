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

    def __init__(self):
        self.weights1 = np.random.rand(16,784) # calculate random weights for layer 1
        self.weights2 = np.random.rand(8,16) # calculate random weights for layer 2
        self.weight_output = np.random.rand(10,8) # calculate random weights for output layer

    def calc_all_layers(self, img:list):

        def __sigmoid(data:list): # element wise sigmoid function
            return (1/(1+(np.exp(-data))))

        def __calc_layer1(): # calculate layer 1
            return __sigmoid(np.dot(self.weights1, img))
        
        def __calc_layer2(): #c calculate layer 2
            return __sigmoid(np.dot(self.weights2, __calc_layer1()))
        
        def __calc_output(): # calculate final output
            return __sigmoid(np.dot(self.weight_output, __calc_layer2()))

        return __calc_output()





if __name__ == "main":

    file = "./data/train.csv"
    df = pd.read_csv(file)
    labels = df['label'].to_numpy()
    df.drop(['label'], axis=1, inplace=True)

    MNIST = data(data={i: df.iloc[i].to_numpy() for i in range(0, len(df.index))}, labels=labels)

    NN = NeuralNetwork()
    for i in range(0, len(MNIST.get_data())):
        print(NN.calc_all_layers(MNIST.get_data()[i]))
