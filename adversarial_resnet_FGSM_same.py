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
                                          batch_size=1,
                                          shuffle=True,
                                          num_workers=20)

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

mean_new=[-0.485, -0.456, -0.406]
std_new=[1/0.229, 1/0.224, 1/0.225]

# Load the ResNet18 model

use_cuda = True

epsilons = [0, 0.1, 0.2, 0.5, 1]

# Define what device is being using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network
model = models.resnet18(pretrained=True)
model.cuda()
# Set the model in evaluation mode
model.eval()

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    perturbed_image = perturbed_image.mul(torch.cuda.FloatTensor(std).view(3,1,1)).add(torch.cuda.FloatTensor(mean).view(3,1,1)) #untransform
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    perturbed_image = perturbed_image.add(torch.cuda.FloatTensor(mean_new).view(3,1,1)).mul(torch.cuda.FloatTensor(std_new).view(3,1,1)) #transform
    # Return the perturbed image
    return perturbed_image

# Accuracy counter
correct = 0
adv_examples = []
orig_examples = []
counter = 0
# Loop over all examples in test set
for data, target in data_loader:
    count = 0
    perturbed_images = []
    original_images = []
    
    for eps in epsilons:
        perturbed_data = None
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
           continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, eps, data_grad)
        perturbed_images.append(perturbed_data.mul(torch.cuda.FloatTensor(std).view(3,1,1)).add(torch.cuda.FloatTensor(mean).view(3,1,1)))
        # Re-classify the perturbed image
        output = model(perturbed_data) 
        data_grad = None
        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() != target.item():
           count += 1
            
    # Save some adv examples for visualization later
    if count == 4:
        data = data.mul(torch.cuda.FloatTensor(std).view(3,1,1)).add(torch.cuda.FloatTensor(mean).view(3,1,1))
        orig_ex = data.squeeze().detach().cpu().numpy()
        orig_examples.append(orig_ex)
        for i in range(5):
            perturbed_data = perturbed_images[i]
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append(adv_ex)     
   
    # Print the progress of evaluation
    counter += 1
    print(counter, "/", len(data_loader))
    del data
    del perturbed_images[:]
    if len(adv_examples) > 20:
       break
       

examples = adv_examples
for j in range(int(len(examples)/len(epsilons))):
    for k in range(len(epsilons)):
        ex = examples[j]
        orig_ex = orig_examples[j]
        ex = torch.from_numpy(ex)
        orig_ex = torch.from_numpy(orig_ex)
        torchvision.utils.save_image(orig_ex, 'orig_{}.png'.format(j), normalize  = True)
        if j == 0:
          torchvision.utils.save_image(ex, 'adv_{}_0.png'.format(k), normalize  = True)
        if j == 1:
          torchvision.utils.save_image(ex, 'adv_{}_1.png'.format(k))
        if j == 2:
          torchvision.utils.save_image(ex, 'adv_{}_2.png'.format(k))
        if j == 3:
          torchvision.utils.save_image(ex, 'adv_{}_3.png'.format(k))
        if j == 4:
          torchvision.utils.save_image(ex, 'adv_{}_4.png'.format(k))

