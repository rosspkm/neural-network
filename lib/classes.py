# contains all classes for Neural Network

class data:
    
    def __init__(self, data:dict, labels:list):
        self.data = data # dict by index as key with row as list in value as numpy array
        self.labels = labels # list of labels for test data for training
    
    def get_data(self):
        return self.data # returns dict of numpy arrays
    
    def get_labels(self):
        return self.labels # returns list of labels