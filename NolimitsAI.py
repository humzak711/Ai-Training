import torch.nn as nn # Import the neural network module
from bs4 import BeautifulSoup # For web scraping and preparing datasets
import requests
import pandas as pd

# Create the neural network
class NeuralNetwork(nn.Module): 
    def __init__(self):
        '''
        Initializes the neural network.
        
        Layers:
        - Input: 2 (features)
        - Hidden Layer 1: 64 neurons with ReLU activation
        - Hidden Layer 2: 32 neurons with ReLU activation
        - Output Layer: 2 neurons with Softmax activation
        '''
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        ''' 
        Takes the input data x and passes it through the layers.
        
        Args:
        - x: Input data
        
        Returns:
        - Output after passing through the neural network layers
        '''
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

# to do
# learn about building ai lol 

# gather training data
# setup training - Define loss functions, optimizers, and any necessary hyperparameters
# setup training loop using requests/beautifulsoup and pandas
# feed the data to the ai
# create CLI 
# evaluate performance
# deploy into the chatbox