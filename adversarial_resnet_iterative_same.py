from __future__ import division
from __future__ import print_function
# Import Libraries
import torch
import time
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import csv
from torch.utils.data.sampler import SubsetRandomSampler

# Get the imagenet_data
imagenet_data = datasets.ImageFolder('val', transform=transforms.Compose([
            transforms.Resize((256)), #resize the image so the shorter side has 255 pixels
            transforms.CenterCrop(224), #get 224 pixel square image
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]))
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=32,
                                          shuffle=True,
                                          num_workers=20)

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

mean_new=[-0.485, -0.456, -0.406]
std_new=[1/0.229, 1/0.224, 1/0.225]

# Load the ResNet18 model

use_cuda = True

epsilons = [0.1, 0.15, 0.2, 0.25]

# Define what device is being using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network
model = models.resnet18(pretrained=True)
model.cuda()
# Set the model in train mode
model.train()


    # Accuracy counter
correct = 0
adv_examples = []
orig_examples = []
counter = 0
alpha = 1
    # Loop over all examples in test set
for data, target in data_loader:
    perturbed_images = []
    perturbed = []
    count = [0] * 32
    original_images = []
    iteration = np.zeros((4, 32))
    num_iteration = []

    for eps in epsilons:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True
     
        output = model(data)
        init_pred = np.squeeze(output.max(1, keepdim=True)[1]) # get the index of the max log-probability
        final_pred = init_pred
 
        s = len(init_pred)

        # Take note of the false ones
        f = []
        for j in range(s):
            if init_pred[j] != target[j]:
               f.append(j)

       # Calculate the loss
        loss = F.cross_entropy(output, target)

       # Zero all existing gradients
        model.zero_grad()

       # Calculate gradients of model in backward pass
        loss.backward()

       # Collect datagrad
        data_grad = data.grad.data

        data_new  = data
        perturbed_data_new = data.detach().clone()
        num_iter = int(round(np.minimum(eps*255 + 4, 1.25*eps*255)))
        num_iteration.append(num_iter)
        physical_counter = 0
        for k in range(num_iter):
           physical_counter += 1

           perturbed_data = perturbed_data_new.detach().clone()

           # Basic Iterative Method
           perturbation = alpha * data_grad.sign()  
           perturbed_data = perturbed_data + perturbation
           perturbed_data = perturbed_data.mul(torch.cuda.FloatTensor(std).view(3,1,1)).add(torch.cuda.FloatTensor(mean).view(3,1,1))
           # Adding clipping to maintain [0,1] range
           perturbed_data = torch.clamp(perturbed_data, 0, 1)
           perturbed_data = perturbed_data.add(torch.cuda.FloatTensor(mean_new).view(3,1,1)).mul(torch.cuda.FloatTensor(std_new).view(3,1,1)) #transform
           perturbed_data_new = perturbed_data.detach().clone()

           # Re-classify the perturbed image
           output = model(perturbed_data_new)

           # Check for success
           final_pred = np.squeeze(output.max(1, keepdim=True)[1]) # get the index of the max log-probability

           # Calculate the loss
           loss = F.cross_entropy(output, target)

           # Zero all existing gradients
           model.zero_grad()

           # Calculate gradients of model in backward pass
           loss.backward()

           # Collect datagrad
           data_grad = data.grad.data


           s = len(final_pred)
           for l in range(s):
               if (l) == len(final_pred) or l > len(final_pred):
                  break
               perturbed_images.append(perturbed_data_new[l, :])
               if final_pred[l] != target[l] and l not in f:
             # Save some adv examples for visualization later
                     count[l] += 1
                     f.append(l)
                     iteration[count[l]-1, l] = physical_counter
                     

    for h in range(32):
            if count[h] == 4:
               me = 0
               for z in range(4):
                   p = int(iteration[z,h])
                   perturbed.append(perturbed_images[int(h+me+(32*(p-1)))])
                   me += num_iteration[z]
               for i in range(len(perturbed)):
                   adv_ex = perturbed[i].squeeze().detach().cpu().numpy()
                   adv_examples.append( (adv_ex) )

               data_new[h, :] = data[h, :].mul(torch.cuda.FloatTensor(std).view(3,1,1)).add(torch.cuda.FloatTensor(mean).view(3,1,1))
               orig_ex = data_new[h, :].squeeze().detach().cpu().numpy()
               orig_examples.append(orig_ex)
    
        
    # Print the progress of evaluation
    counter += 1
    print(counter, "/", len(data_loader))
    del perturbed_images
    if len(orig_examples) > 3:
       break

print(iteration)
print(count)

examples = adv_examples
for j in range(len(orig_examples)):
    orig_ex = orig_examples[j]
    orig_ex = torch.from_numpy(orig_ex)
    torchvision.utils.save_image(orig_ex, 'Iorig_{}.png'.format(j), normalize  = True)
    for k in range(4):
        ex = examples[4*j + k]
        ex = torch.from_numpy(ex)

        torchvision.utils.save_image(ex, 'Iadv_{}_{}.png'.format(j, k), normalize  = True)


