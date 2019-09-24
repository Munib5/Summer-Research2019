# Import Libraries
import torch
from torch.utils.data.dataset import Dataset, TensorDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import csv

class CSVDataset(Dataset):
    def __init__(self, csv_path, transforms = None):
        self.data = pd.read_csv(csv_path, header=0, names=['age', 'sex', 'test_time', 'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP', 'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE', 'total_UPDRS'])
        self.target = self.data.iloc[:, 19]

    def __len__(self):
        return len(self.data)-1

    def __getitem__(self, index):
        target = torch.from_numpy(np.array(self.target.iloc[index]))
        x = torch.from_numpy(np.array(self.data.iloc[index, 1:19]))
        return (x, target)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout = nn.Dropout(0.2) #to prevent overfitting drop out units
        self.fc1 = nn.Linear(18, 84)
        self.fc2 = nn.Linear(84, 16)
        self.fc4 = nn.Linear(16, 1)
        
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.fc4(x)
        return x

device = torch.device('cpu')
model = Net()
optimizer = torch.optim.Adam(model.parameters(), 1e-4)

checkpoint = torch.load("/home/munib/Desktop/regression/model_epoch.pt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model_dataset = CSVDataset("/home/munib/Desktop/regression/model.csv")

model_loader = torch.utils.data.DataLoader(model_dataset, batch_size=1, shuffle=False)

input_values = []
output_values = []

model.eval()
counter = 0
    # Tell torch not to calculate gradients
with torch.no_grad():
 for inputs, labels in model_loader:
            # Move to device
   inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
   output = model(inputs.float())
   print(inputs)
   input_values.append(inputs.float())
   output_values.append(output)
            
            # Print the progress of evaluation
   counter += 1
   print(counter, "/", len(model_loader))

csvfile = "/home/munib/Desktop/regression/input_values.txt"
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in input_values:
        writer.writerow([val])
csvfile = "/home/munib/Desktop/regression/output_values.txt"
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in output_values:
        writer.writerow([val])
