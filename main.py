# Original author: Morgan McKinney 4/2021

import numpy as np
import torch
from torch import nn
import torch.nn.functional as f
from torch.utils.data import TensorDataset, DataLoader


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
        # hidden_state = self.init_hidden()
        output, hidden_state = self.rnn(x)
        # Use this to deal with the extra dimension from having a batch
        output = output.contiguous().view(-1, self.hidden_size)
        output = self.fc(output)
        return output, hidden_state

    # def init_hidden(self):
        # Remember, (row, BATCH, column)
        # hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
        # return hidden


# Create one-hot vector
def create_one_hot(sequence, v_size):
    # Define a matrix of size vocab_size containing all 0's
    # Dimensions: Batch Size x Sequence Length x Vocab Size
    # Have to do this even if your batch size is 1
    encoding = np.zeros((1, len(sequence), v_size), dtype=np.float32)
    for i in range(len(sequence)):
        encoding[0, i, sequence[i]] = 1
    return encoding  # Return one-hot sequence


# Prediction
def predict(model, character):
    characterInput = np.array([charInt[c] for c in character])
    characterInput = create_one_hot(characterInput, vocab_size)
    characterInput = torch.from_numpy(characterInput).cuda()
    out, hidden = model(characterInput)

    prob = nn.functional.softmax(out[-1], dim=0).data
    character_index = torch.max(prob, dim=0)[1].item()

    return intChar[character_index], hidden


# Sample
def sample(model, out_len, start='QUEEN:'):
    characters = [ch for ch in start]
    currentSize = out_len - len(characters)
    for i in range(currentSize):
        character, hidden_state = predict(model, characters)
        characters.append(character)

    return ''.join(characters)


# Initialize variables
input_sequence = []
target_sequence = []
sentences = []
# Hyperparamters
epochs = 1
print_frequency = 1000  # Loss print frequency
batch = 2

# Read data
file = open("tiny-shakespeare.txt", "r").read()
# Set up vocabulary
characters = list(set(file))
intChar = dict(enumerate(characters))
charInt = {character: index for index, character in intChar.items()}
vocab_size = len(charInt)
# Split corpus into segments
segments = [file[pos:pos+42] for pos, i in enumerate(list(file)) if pos % 42 == 0]
# Combine every 4 segments, of length 42, into length 168
new_segment = ""
for i in range(len(segments)):
    new_segment += segments[i]
    if i % 4 == 3:
        sentences.append(new_segment)
        new_segment = ""
# Set up input and target sequences
for i in range(len(sentences)):
    input_sequence.append(sentences[i][:-1])
    target_sequence.append(sentences[i][1:])
# Replace characters with integer values
for i in range(len(sentences)):
    input_sequence[i] = [charInt[character] for character in input_sequence[i]]
    target_sequence[i] = [charInt[character] for character in target_sequence[i]]

# Batch data
training = TensorDataset(torch.FloatTensor(input_sequence), torch.FloatTensor(target_sequence))
trainLoader = DataLoader(training, batch_size=batch)  # Shuffle?

# Set up model, loss, and optimizers
model = RNNModel(vocab_size, vocab_size, 300, 2)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Train
model.cuda()
print(next(model.parameters()).is_cuda)
for epoch in range(epochs):
    print("Epoch:", epoch)
    # for x, y in trainLoader:
    for i in range(len(input_sequence)):
        optimizer.zero_grad()
        # Create input as a tensor
        input_tensor = torch.from_numpy(create_one_hot(input_sequence[i], vocab_size))  # We don't need this once fixed
        # Create target; cross entropy loss uses integer thus no need for one-hots
        target_tensor = torch.Tensor(target_sequence[i])
        # Train using GPU
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()
        output, hidden = model(input_tensor)
        lossValue = loss(output, target_tensor.view(-1).long())
        lossValue.backward()
        optimizer.step()
        if i % print_frequency == 0:
            print("Loss: {:.4f}".format(lossValue.item()))

# Output
print(sample(model, 50))
