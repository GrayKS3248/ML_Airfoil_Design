# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 18:55:56 2021

@author: GKSch
"""

import torch
import torch.nn as nn

class NN(nn.Module):
    
    def __init__(self, n_panels):
        
        # Initialize inherited class
        super(NN, self).__init__()
        
        size = n_panels + 1
        
        #Initialize the encoding linear layers
        self.fc1 = nn.Linear(size, 8*size)
        self.fc2 = nn.Linear(8*size, 8*size)
        self.fc3 = nn.Linear(8*size, 7)

        #Initialize the decoding linear layers
        self.t_fc1 = nn.Linear(7, 8*size)
        self.t_fc2 = nn.Linear(8*size, 8*size)
        self.t_fc3 = nn.Linear(8*size, size)

    def forward(self, x):
        #Feed-forward x
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        
        #Return x
        return x
    
    def backward(self, x):
        #Feed-forward x
        x = torch.tanh(self.t_fc1(x))
        x = torch.tanh(self.t_fc2(x))
        x = self.t_fc3(x)
        
        #Return x
        return x  