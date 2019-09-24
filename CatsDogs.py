# Import Libraries
import torch
import time
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import csv
from torch.utils.data.sampler import SubsetRandomSampler

transform = transforms.Compose([ #compose do add up multiple transforms
    transforms.Resize((255)), #resize the image so the shorter side has 255 pixels
    transforms.CenterCrop(224), #get 224 pixel square image
    transforms.RandomHorizontalFlip(), #flip horizontally because the image could have been taken from either left or right
    transforms.RandomRotation(20), #the angle of the image taken could be different
    transforms.ToTensor(), #transform the dataset into a tensor for pytorch to use
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]) #Define a transform object to perform the preprocessing step

train_data = datasets.ImageFolder('train', transform=transform)

valid_size = 0.25
test_size = 0.25

#For test
num_data = len(train_data)
indices_data = list(range(num_data))
np.random.shuffle(indices_data)
split_tt = int(np.floor(test_size * num_data))
train_idx, test_idx = indices_data[split_tt:], indices_data[:split_tt]

#For Valid
num_train = len(train_idx)
indices_train = list(range(num_train))
np.random.shuffle(indices_train)
split_tv = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices_train[split_tv:],indices_train[:split_tv]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
test_sampler = SubsetRandomSampler(test_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

#Dataloader is able to spit out random samples of our data in batches which makes training more efficient
train_loader = torch.utils.data.DataLoader(train_data, sampler = train_sampler, batch_size=32,num_workers=20, pin_memory = True)
test_loader = torch.utils.data.DataLoader(train_data, sampler = test_sampler, batch_size=32,num_workers=20, pin_memory = True)
valid_loader = torch.utils.data.DataLoader(train_data, sampler = valid_sampler, batch_size=32,num_workers=20, pin_memory = True) #workers tells you how many separate assignments to RAM get made at one time


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(3, 16, 5) #3 for color image, less layers is more economical, the kernel size is 5
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2) #kernel and stride
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.conv4 = nn.Conv2d(64, 32, 5)
        self.dropout = nn.Dropout(0.2) #to prevent overfitting drop out units
        self.fc1 = nn.Linear(32*10*10, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 1)
        self.fc4 = nn.Sigmoid()
        
    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = self.pool(nn.functional.relu(self.conv4(x)))
        x = x.view(-1, 32*10*10)
        x = self.dropout(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(nn.functional.relu(self.fc2(x)))
        x = self.fc3(x)
        x = self.fc4(x)
        x = x.squeeze()
        return x

# create a complete CNN
model = Net()

#GPU is usually faster to use so check first if it can be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to the device specified above
model.to(device)

# Set the error function as binary cross-entropy
criterion = nn.BCELoss()

# Set the optimizer function as Adam
optimizer = optim.Adam(model.parameters())

def acc(output, target):
    """Computes the accuracy for multiple binary predictions"""
    pred = output >= 0.5
    truth = target >= 0.5
    acc = (pred == truth).sum().item() / 32
    return acc

acc_list = []
trainloss_list = []
validloss_list = []
trainingacc_list = []
epochs = 500 #how many iterations of updates and training
for epoch in range(epochs):
    train_loss = 0
    val_loss = 0
    accuracy = 0 #error
    trainingacc = 0
    # Training the model
    model.train()
    counter = 0

    for inputs, labels in train_loader:
        # Move to device
        inputs, labels = inputs.to(device), labels.to(device)

        # Clear optimizers
        optimizer.zero_grad()


        # Forward pass
        output = model(inputs)
        labels = labels.type(torch.FloatTensor) #for loss function to work the target must be float

        loss = criterion(output, labels)

        # Calculate gradients (backpropagation)
        loss.backward()

        # Adjust parameters based on gradients
        optimizer.step()

        # Add the loss to the training set's running loss
        train_loss += loss.item()*inputs.size(0)
 
        trainingacc += acc(output, labels)

        # Print the progress of training
        counter += 1
        print(counter, "/", len(train_loader))

    # Evaluating the model
    model.eval()
    counter = 0
    # Tell torch not to calculate gradients
    with torch.no_grad():
        for inputs, labels in valid_loader:
            # Move to device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            output = model(inputs)
            labels = labels.type(torch.FloatTensor) #for loss function to work the target must be float
            # Calculate Loss
            valloss = criterion(output, labels)

            # Add loss to the validation set's running loss
            val_loss += valloss.item()*inputs.size(0)

            # Calculate the mean (get the accuracy for this batch)
            # and add it to the running accuracy for this epoch
            accuracy += acc(output, labels)
            
            # Print the progress of evaluation
            counter += 1
            print(counter, "/", len(valid_loader))
    
    # Get the average loss for the entire epoch
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = val_loss/len(valid_loader.dataset)

    # Print out the information
    print('Accuracy: ', accuracy/len(valid_loader))
    acc_list.append(accuracy/len(valid_loader))
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
    trainloss_list.append(train_loss)
    validloss_list.append(valid_loss)
    trainingacc_list.append(trainingacc/len(train_loader))

csvfile = "/home/munib/catvdog/accuracy_values.txt"
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in acc_list:
        writer.writerow([val])
csvfile = "/home/munib/catvdog/trainingaccuracy_values.txt"
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in trainingacc_list:
        writer.writerow([val])
csvfile = "/home/munib/catvdog/trainloss_values.txt"
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in trainloss_list:
        writer.writerow([val])
csvfile = "/home/munib/catvdog/validloss_values.txt"
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in validloss_list:
        writer.writerow([val])
