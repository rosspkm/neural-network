# neural network start
import numpy as np
import pandas as pd

class data:

    def __init__(self, data:Dataframe, labels:Dataframe):
        self.data = data
        self.labels = labels
    
    def get_data(self):
        return self.data
    
    def get_labels(self):
        return self.labels


class NeuralNetwor:

    def __init__(self, x:list):
        self.input = x
        

data = "./data/train.csv"

df = pd.read_csv(data)
labels = pd.DataFrame(df.loc[: , 'label'])
df.drop(['label'], axis=1, inplace=True)


MNIST = data(data=df, labels=labels)

# print(len(df.index))
# for i in range(0, len(df.index)):
#     print(df.loc[i].values)

