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

grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

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

epsilons = [0.001, 0.003, 0.0075, 0.015, 0.03, 0.075, 0.5, 0.7, 0.85]

# Define what device is being using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network
model = models.resnet18(pretrained=True)
model.cuda()
# Set the model in train mode
model.train()

def test(model, device, data_loader, epsilon):


    # Accuracy counter
    correct = 0
    adv_examples = []
    orig_examples = []
    counter = 0
    alpha = 1
    # Loop over all examples in test set
    for data, target in data_loader:

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
        num_iter = int(round(np.minimum(epsilon*10 + 4, 1.25*epsilon*10)))

        for k in range(num_iter):

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
               if final_pred[l] != target[l] and l not in f:
             # Save some adv examples for visualization later
                  if len(adv_examples) < 5:
                     f.append(l)
                     perturbed_data_new[l, :] = perturbed_data_new[l, :].mul(torch.cuda.FloatTensor(std).view(3,1,1)).add(torch.cuda.FloatTensor(mean).view(3,1,1))
                     adv_ex = perturbed_data_new[l, :].squeeze().detach().cpu().numpy()
                     data_new[l, :] = data[l, :].mul(torch.cuda.FloatTensor(std).view(3,1,1)).add(torch.cuda.FloatTensor(mean).view(3,1,1))
                     orig_ex = data_new[l, :].squeeze().detach().cpu().numpy()
                     adv_examples.append( (init_pred[l], final_pred[l], adv_ex) )
                     orig_examples.append(orig_ex)

        correct += final_pred.eq(target).sum()
        for g in range(s):
             # Special case for saving 0 epsilon examples
              if (epsilon == 0) and (len(adv_examples) < 5) and g not in f:
                 perturbed_data_new[g, :] = perturbed_data_new[g, :].mul(torch.cuda.FloatTensor(std).view(3,1,1)).add(torch.cuda.FloatTensor(mean).view(3,1,1))
                 adv_ex = perturbed_data_new[g, :].squeeze().detach().cpu().numpy()
                 data_new[g, :] = data[g, :].mul(torch.cuda.FloatTensor(std).view(3,1,1)).add(torch.cuda.FloatTensor(mean).view(3,1,1))
                 orig_ex = data_new[g, :].squeeze().detach().cpu().numpy()
                 adv_examples.append( (init_pred[g], final_pred[g], adv_ex) )
                 orig_examples.append(orig_ex)
        
        # Print the progress of evaluation
        counter += 1
        print(counter, "/", len(data_loader))

    # Calculate final accuracy for this epsilon
    final_acc = float(correct)/float(32*float(len(data_loader)))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, 32*len(data_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples, orig_examples

# Run the attack
accuracies = []
examples = []
orig_examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex, orig_ex = test(model, device, data_loader, eps)
    accuracies.append(acc)
    examples.append(ex)
    orig_examples.append(orig_ex)

csvfile = "/newhd/munib/ILSCVR12/Accuracy_Iterative.txt"
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in accuracies:
        writer.writerow([val])

#original = []
#adversarial = []
#for i in range(len(epsilons)):
 #   for j in range(len(examples[i])):
  #      orig,adv,ex = examples[i][j]
   #     orig_ex = orig_examples[i][j]
    #    original.append(orig)
     #   adversarial.append(adv)
      #  ex = torch.from_numpy(ex)
       # orig_ex = torch.from_numpy(orig_ex)
   #     if j == 0:
    #      torchvision.utils.save_image(ex, 'Iadv_{}_0.png'.format(i))
     #     torchvision.utils.save_image(orig_ex, 'Iorig_{}_0.png'.format(i))
      #  if j == 1:
       #   torchvision.utils.save_image(ex, 'Iadv_{}_1.png'.format(i))
        #  torchvision.utils.save_image(orig_ex, 'Iorig_{}_1.png'.format(i))
     #   if j == 2:
      #    torchvision.utils.save_image(ex, 'Iadv_{}_2.png'.format(i))
       #   torchvision.utils.save_image(orig_ex, 'Iorig_{}_2.png'.format(i))
  #      if j == 3:
   #       torchvision.utils.save_image(ex, 'Iadv_{}_3.png'.format(i))
    #      torchvision.utils.save_image(orig_ex, 'Iorig_{}_3.png'.format(i))
     #   if j == 4:
     #     torchvision.utils.save_image(ex, 'Iadv_{}_4.png'.format(i))
      #    torchvision.utils.save_image(orig_ex, 'Iorig_{}_4.png'.format(i))

