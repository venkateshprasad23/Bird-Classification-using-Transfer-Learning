#!/usr/bin/env python
# coding: utf-8

# # Venkatesh Prasad Venkataramanan PID : A53318036 

get_ipython().run_line_magic('matplotlib', 'inline')

################################################## Importing required libraries #####################################################################

import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data as td
import torchvision as tv
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import nntools as nt


############################################## Using GPU for training ##################################################################################

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

############################################## Setting the directory of the dataset ################################################################## 

dataset_root_dir = '/datasets/ee285f-public/caltech_ucsd_birds'

############################################## A class which crops the images using bounding boxes, and then resizes and normalizes them ############# 

class BirdsDataset(td.Dataset):

    def __init__(self, root_dir, mode="train", image_size=(224, 224)):
        super(BirdsDataset, self).__init__()
        self.image_size = image_size
        self.mode = mode
        self.data = pd.read_csv(os.path.join(root_dir, "%s.csv" % mode))
        self.images_dir = os.path.join(root_dir, "CUB_200_2011/images")

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "BirdsDataset(mode={}, image_size={})".                format(self.mode, self.image_size)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir,                                 self.data.iloc[idx]['file_path'])
        bbox = self.data.iloc[idx][['x1', 'y1', 'x2', 'y2']]
        img = Image.open(img_path).convert('RGB')
        img = img.crop([bbox[0], bbox[1], bbox[2], bbox[3]])
        transform = tv.transforms.Compose([
            tv.transforms.Resize(self.image_size),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((.5, .5, .5), (.5, .5, .5))
            
            ])
        x = transform(img)
        d = self.data.iloc[idx]['class']
        return x, d

    def number_of_classes(self):
        return self.data['class'].max() + 1


############################################## A function to display the images by first transferring to CPU and then converting into a Numpy object ####

def myimshow(image, ax=plt):
    image = image.to('cpu').numpy()
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
    image = (image + 1) / 2
    image[image < 0] = 0
    image[image > 1] = 1
    h = ax.imshow(image)
    ax.axis('off')
    return h

############################################## Instantiating the train_set as an object of the class Birds Dataset and using DataLoader class from Pytorch to shuffle ### 

train_set = BirdsDataset(dataset_root_dir)
train_loader = td.DataLoader(train_set, batch_size=16, shuffle=True, pin_memory=True)

############################################## Instantiating the val_set as an object of the class Birds Dataset and setting shuffle to False ###########################

val_set = BirdsDataset(dataset_root_dir, mode='val')
val_loader = td.DataLoader(val_set, batch_size=16, pin_memory=True)

############################################## Defining a class which inherits from the abstract class Neural Network. Note that NNClassifier is still abstract ######### 

class NNClassifier(nt.NeuralNetwork):

    def __init__(self):
        super(NNClassifier, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def criterion(self, y, d):
        return self.cross_entropy(y, d)

############################################## Downloading the VGG16 Network ############################################################################################

vgg = tv.models.vgg16_bn(pretrained=True)

############################################## Defining a class which inherits from NNClassifier ######################################################################## 

class VGG16Transfer(NNClassifier):

    def __init__(self, num_classes, fine_tuning=False):
        super(VGG16Transfer, self).__init__()
        vgg = tv.models.vgg16_bn(pretrained=True)
        for param in vgg.parameters():
            param.requires_grad = fine_tuning
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.classifier = vgg.classifier
        num_ftrs = vgg.classifier[6].in_features
        self.classifier[6] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        f = torch.flatten(x, 1)
        y = self.classifier(f)
        return y

############################################## Instantiating an object of the class VGG16Transfer ########################################################################

num_classes = train_set.number_of_classes()
vgg16transfer = VGG16Transfer(num_classes)

############################################## Defining a class which inherits from StatsManager of NNTools, to print accuracy ###########################################

class ClassificationStatsManager(nt.StatsManager):

    def __init__(self):
        super(ClassificationStatsManager, self).__init__()

    def init(self):
        super(ClassificationStatsManager, self).init()
        self.running_accuracy = 0

    def accumulate(self, loss, x, y, d):
        super(ClassificationStatsManager, self).accumulate(loss, x, y, d)
        _, l = torch.max(y, 1)
        self.running_accuracy += torch.mean((l == d).float())

    def summarize(self):
        loss = super(ClassificationStatsManager, self).summarize()
        accuracy = 100 * self.running_accuracy / self.number_update
        return {'loss': loss, 'accuracy': accuracy}

############################################## Defining parameters for experiment with VGG16 #############################################################################

lr = 1e-3
net = VGG16Transfer(num_classes)
net = net.to(device)
adam = torch.optim.Adam(net.parameters(), lr=lr)
stats_manager = ClassificationStatsManager()
exp1 = nt.Experiment(net, train_set, val_set, adam, stats_manager,
               output_dir="birdclass1", perform_validation_during_training=True)

############################################## Function to plot training and evaluation losses ###################################################################################

def plot(exp, fig, axes):
    axes[0].clear()
    axes[1].clear()
    axes[0].plot([exp.history[k][0]['loss'] for k in range(exp.epoch)],
                 label="training loss")
    
    axes[0].plot([exp.history[k][1]['loss'] for k in range(exp.epoch)],
                 label="evaluation loss")
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(('training loss', 'evaluation loss'))
    
    axes[1].plot([exp.history[k][0]['accuracy'] for k in range(exp.epoch)],
                 label="training accuracy")
    
    axes[1].plot([exp.history[k][1]['accuracy'] for k in range(exp.epoch)],
                 label="evaluation accuracy")
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend(('training accuracy', 'evaluation accuracy'), loc='lower right')
    plt.tight_layout()
    fig.canvas.draw()

############################################## Plotting the statistics of experiment 1 ###########################################################################################

fig, axes = plt.subplots(ncols=2, figsize=(7, 3))
exp1.run(num_epochs=20, plot=lambda exp: plot(exp, fig=fig, axes=axes))

############################################## Instantiating an object of the class ResNet18Transfer ############################################################################

resnet = tv.models.resnet18(pretrained=True)

class Resnet18Transfer(NNClassifier):

    def __init__(self, num_classes, fine_tuning=False):
        super(Resnet18Transfer, self).__init__()
        resnet = tv.models.resnet18(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = fine_tuning
        self.classifier = resnet
        num_ftrs = resnet.fc.in_features
        self.classifier.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):     
        y = self.classifier(x)
        return y

resnet18transfer = Resnet18Transfer(num_classes)

############################################## Parameters for experiment 2 ###################################################################################################

lr2 = 1e-3
net2 = Resnet18Transfer(num_classes)
net2 = net2.to(device)
adam2 = torch.optim.Adam(net2.parameters(), lr=lr2)
stats_manager2 = ClassificationStatsManager()
exp2 = nt.Experiment(net2, train_set, val_set, adam2, stats_manager2,
               output_dir="birdclass2", perform_validation_during_training=True)

fig, axes = plt.subplots(ncols=2, figsize=(7, 3))
exp2.run(num_epochs=20, plot=lambda exp: plot(exp, fig=fig, axes=axes))


########################################### Evaluating performances of the two experiments ##################################################################################


exp1.evaluate()
exp2.evaluate()


############################################################################################################################################################################
