#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */ImageClassifier/model.py
#                                                                             
# PROGRAMMER: Michael J Russell Jr
# DATE CREATED: 4/20/2021
# REVISED DATE: 4/24/2021
# PURPOSE: Creates and validates a model 
from collections import OrderedDict
from torchvision import models
from torch import nn, optim
import torch



def create_model(pretrained_model, dropout, hidden_layer, learning_rate):
    """
    Creates the model with the dropout, hidden layer, learninf rate, and pytorch model
    that will be trained or loaded from a checkpoint file.
    Parameters:
     pretrained_model - The pyTorch architecture to use for the model
     dropout - The droup out precent to be used for the model
     hidden_layer - Sets the amount of hidden layers for the model
     learning_rate - The learninf rate of the model
    Returns:
     model - The loaded trained model
     criterion - The loaded criterion
     optimizer - The loaded optimizer
    """
    
    print("Creating Model with parameters: Arch: {} Dropout: {} hidden layer: {} learning_rate: {}".format(pretrained_model, dropout,                                                                                                                  hidden_layer, learning_rate))
    
    model = models.__dict__[pretrained_model](pretrained=True)
    classifier_name, old_classifier = model._modules.popitem()

    if type(old_classifier) == nn.Linear:
        classifier_input_size = old_classifier.in_features
    else:
        classifier_input_size = old_classifier[0].in_features

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                                ("fc1", nn.Linear(classifier_input_size, hidden_layer)),
                                ("relu", nn.ReLU()),
                                ("dropout", nn.Dropout(p=dropout)),
                                ("fc2", nn.Linear(hidden_layer, 102)),
                                ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.add_module(classifier_name, classifier)
    optimizer = optim.Adam(model._modules.get(classifier_name).parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    return model, criterion, optimizer

def validate_model(model, device, val_data):
    """
    Validates the model with the dataset
    Parameters:
     model - The trained model to validate the data with
     device - The device used to validate the data
     val_data - The dataset used to validate the model
    Returns:
     none
    """
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_data:
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model(inputs)

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    print(f"Validation Accuracy: {(accuracy/len(val_data))*100:.3f}%")
    return accuracy/len(val_data)