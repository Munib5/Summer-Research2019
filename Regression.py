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

transformations = transforms.Compose([transforms.ToTensor()])

#data.columns = ['age', 'sex', 'test_time', 'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP', 'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE', 'total_UPDRS']

# Create Dataset
class CSVDataset(Dataset):
    def __init__(self, csv_path, transforms = None):
        self.data = pd.read_csv(csv_path, header=0, names=['age', 'sex', 'test_time', 'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP', 'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE', 'total_UPDRS'])
        self.target = self.data.iloc[:, 19]
        self.transforms = transformations

    def __len__(self):
        return len(self.data)-1

    def __getitem__(self, index):
        target = torch.from_numpy(np.array(self.target.iloc[index]))
        x = torch.from_numpy(np.array(self.data.iloc[index, 1:19]))
        return (x, target)

test_dataset = CSVDataset("/home/munib/regression/parkinsons_test.csv", transformations) #0.3 split in separate python script
train_dataset = CSVDataset("/home/munib/regression/parkinsons_train.csv", transformations)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=20, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=20, shuffle=True)

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

# create a complete Network
model = Net()

#GPU is usually faster to use so check first if it can be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to the device specified above
model.to(device)

# Set the error function as binary cross-entropy
criterion = nn.MSELoss()

# The smaller the learning rate the more computationally demanding the software is
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

trainloss_list = []
validloss_list = []
epochs = 500 #how many iterations of updates and training
for epoch in range(epochs):
    train_loss = 0
    val_loss = 0
    
    # Training the model
    model.train()
    counter = 0
    for inputs, labels in train_loader:
        # Move to device
        inputs, labels = inputs.to(device), labels.to(device)
        print(inputs.size())
        # Clear optimizers
        optimizer.zero_grad()

        # Forward pass
        output = model(inputs.float())

        # Loss
        loss = criterion(output, labels.float())

        # Calculate gradients (backpropagation)
        loss.backward()

        # Adjust parameters based on gradients
        optimizer.step()

        # Add the loss to the training set's running loss
        train_loss += loss.item()*inputs.size(0)

        # Print the progress of training
        counter += 1
        print(counter, "/", len(train_loader))
        
    # Evaluating the model
    model.eval()
    counter = 0
    # Tell torch not to calculate gradients
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move to device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            output = model(inputs.float())

            # Calculate Loss
            valloss = criterion(output, labels.float())

            # Add loss to the validation set's running loss
            val_loss += valloss.item()*inputs.size(0)

            
            # Print the progress of evaluation
            counter += 1
            print(counter, "/", len(test_loader))
    
    # Get the average loss for the entire epoch
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = val_loss/len(test_loader.dataset)

    # Print out the information
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
    trainloss_list.append(train_loss)
    validloss_list.append(valid_loss)

csvfile = "/home/munib/regression/trainloss_values.txt"
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in trainloss_list:
        writer.writerow([val])
csvfile = "/home/munib/regression/validloss_values.txt"
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in validloss_list:
        writer.writerow([val])

torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, "/home/munib/regression/model_epoch.pt")
