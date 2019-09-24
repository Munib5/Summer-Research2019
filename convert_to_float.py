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

def concatenate_list_data(list):
    result= ''
    for element in list:
        result += str(element)
    return result

float_output = []
with open("output_values.txt") as csv_file:
    csv_reader = csv.reader(csv_file, lineterminator='\n')
    line_count = 0
    for row in csv_reader:
       l = []
       count = 0
       for t in row[0]:
         if t.isdigit():
             l.append(int(t))
             count += 1
             if count == 2:
               l.append('.')
       float_output.append(concatenate_list_data(l))

csvfile = "/home/munib/Desktop/regression/float_output_values.txt"
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in float_output:
        writer.writerow([val])
