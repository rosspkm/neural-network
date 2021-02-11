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
    
    #def get_row(self, index:int):




class NeuralNetwork:

    def __init__(self, x:list):
        self.input = x
        

file = "./data/train.csv"
df = pd.read_csv(file)
labels = df['label'].to_numpy()
df.drop(['label'], axis=1, inplace=True)

MNIST = data(data={i: df.iloc[i].to_numpy() for i in range(0, len(df.index))}, labels=labels)

# print(len(df.index))
# for i in range(0, len(df.index)):
#     print(df.loc[i].values)

