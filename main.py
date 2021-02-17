# neural network start
import numpy as np
import pandas as pd
from lib import classes

class NeuralNetwork:

    def __init__(self, input_size:int):
        self.weights1 = np.random.rand(16,input_size) # calculate random weights for layer 1
        self.weights2 = np.random.rand(8,16) # calculate random weights for layer 2
        self.weight_output = np.random.rand(10,8) # calculate random weights for output layer

        self.z_output = None
        self.layer1_output = None
        self.layer2_output = None
        self.output_layer_output = None

    def calc_all_layers(self, img:list):
        # def __sigmoid(data:list): # element wise sigmoid function
        #     return (1/(1+(np.exp(-data))))

        def __relu(data:list):
            return np.maximum(0,data) 

        def __calc_layer1(): # calculate layer 1
            self.layer1_output = __relu(data=np.dot(self.weights1, img))
            return self.layer1_output
        
        def __calc_layer2(): # calculate layer 2
            self.layer2_output = __relu(data=np.dot(self.weights2, __calc_layer1()))
            return self.layer2_output
        
        def __calc_output(): # calculate final output
            self.z_output = np.dot(self.weight_output, __calc_layer2())
            self.output_layer_output = __relu(data=self.z_output)
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
        print(y-z)



        print(len(np.array([0 if ele <= 0 else ele for ele in self.z_output])))
        print(len(self.layer2_output))
        return (y-z)

    def layer2_gradient(self, target_cat:int, img:list):
        temp = self.output_layer_gradient(target_cat=target_cat, img=img) # dont use to calculate anything
        output = np.zeros(8,16)
        for row in range(0, len(self.weights2)):
            for col in range(0, len(self.weights2[row])):
                if self.layer2_output[row][col] == 0:
                    output[row][col] = 0
                else:
                    output[row][col] = self.weights2[row][col]

        return output

    def layer1_gradient(self, target_cat:int, img:list):
        temp = self.output_layer_gradient(target_cat=target_cat, img=img) # dont use to calculate anything
        output = np.zeros(8,16)
        for row in range(0, len(self.weights1)):
            for col in range(0, len(self.weights1[row])):
                if self.layer1_output[row][col] == 0:
                    output[row][col] = 0
                else:
                    output[row][col] = self.weights1[row][col]

        return output 



    # c0 - 1/2(ouput-target)**2
    # 2*1/2(output-target) = output-target

    # zL - dot product of (weights_output)(__calc_layer2) (self.z_output)
    # wL - output layer weights (self.weights_output)
    # aL - output layer output(self.output_layer_output)

    # del(c0)/del(wL) - is the derivative of the outputlayer_weights with 
    # respect of the cost function (output_layer_gradient(may not have it right yet))

    # del(zL)/del(wL) - is the derivative of output layer weights with respect to the output layer output
    # del(aL)/del(zL) - is the derivative of zl with respect to al
    # del(c0)/del(aL) - is the derivative of the output layer output with respects to the cost function

    # output_layer_weights_gradient = 
    # del(c0)/del(aL) * del(c0)/del(aL) * del(zL)/del(wL) 
    # del(c0)/del(aL) = output-target (output_layer_gradient (y-z))
    # del(aL)/del(zL) = 0 if less than 0 else 1
    # del(zL)/del(wL) = 

    # 10,8
    # 10,1
    # 10,1
    # 

    # 

    # aL-1 = [y1, y2, y3]
    # wL   = [x1, x2, x3]
    #        [x4, x5, x6]
    # zl =   [y1*x1 + y2*x2 + y3*x3] = z1
    #        [y1*x4 + y2*x5 + y3*x3] = z2
    # 
    # del(c0)/del(aL) = [o1, o2, o3, o4, o5, o6, o7, o8, o8, o9]
    # del(aL)/del(zL)       

# make this a function eventually
file = "./data/train.csv"
df = pd.read_csv(file)
labels = df['label'].to_numpy()
df.drop(['label'], axis=1, inplace=True)

MNIST = classes.data(data={i: df.iloc[i].to_numpy() for i in range(0, len(df.index))}, labels=labels)

NN = NeuralNetwork(len(MNIST.get_data()[0])+1)
#for i in range(0, len(MNIST.get_data())):

img = np.concatenate([[1], MNIST.get_data()[1]])
NN.output_layer_gradient(MNIST.get_labels()[0], img)