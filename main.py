# Original author: Morgan McKinney 4/2021

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as f
from nltk import tokenize


# Define recurrent neural network model
class RNNModel(nn.Module):
    # Define layer information
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Default batch index is 1 and not 0
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


def create_one_hot(sequence, v_size):
    # Define a matrix of size vocab_size containing all 0's
    # Dimensions: Batch Size x Sequence Length x Vocab Size
    # Have to do this even if your batch size is 1
    encoding = np.zeros((1, len(sequence), v_size), dtype=np.float32)
    for i in range(len(sequence)):
        encoding[0, i, sequence[i]] = 1
    return encoding  # Return one-hot sequence


# Initialize variables
input_sequence = []
target_sequence = []

# Read data
file = open("tiny-shakespeare.txt", "r").read()
# Set up vocabulary
characters = list(set(file))
intChar = dict(enumerate(characters))
charInt = {character: index for index, character in intChar.items()}
vocab_size = len(charInt)
# Split corpus into segments
sentences = tokenize.sent_tokenize(file)

# Set up input and target sequences
for i in range(len(sentences)):
    input_sequence.append(sentences[i][:-1])
    target_sequence.append(sentences[i][1:])
# Replace characters with integer values
for i in range(len(sentences)):
    input_sequence[i] = [charInt[character] for character in input_sequence[i]]
    target_sequence[i] = [charInt[character] for character in target_sequence[i]]

# Set up training and test sets
