# Original author: Morgan McKinney 4/2021

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as f


# Define recurrent neural network model
class RNNModel(nn.Module):
    # Define layer information
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Default batch index is 1 and not 0.
        # batch_first = true -> (batch, sequence, word)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # This is to make sure that our output dimension is correct
        self.fc = nn.Linear(hidden_size, output_size)

    # Define how inputs translate into outputs
    def forward(self, x):
        hidden_state = self.init_hidden()
        output, hidden_state = self.rnn(x, hidden_state)
        # Use this to deal with the extra dimension from having a batch
        output = output.contiguous().view(-1, self.hidden_size)
        output = self.fc(output)
        return output, hidden_state

    def init_hidden(self):
        # Remember, (row, BATCH, column)
        hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
        return hidden


# Initialize variables
train_data = []
test_data = []
class_label_index = 0

# Read data
# Split corpus into segments
# Convert every character in vocabulary into one-hot vector

# Set up training and test sets
