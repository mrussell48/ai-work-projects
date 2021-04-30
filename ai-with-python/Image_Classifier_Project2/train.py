#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */ImageClassifier/train.py
#                                                                             
# PROGRAMMER: Michael J Russell Jr
# DATE CREATED: 4/20/2021
# REVISED DATE: 4/24/2021
# PURPOSE: To train a CCN model 

import argparse
import torch

from torchvision import datasets, transforms
from workspace_utils import active_session
from model import create_model, validate_model

def load_data(data_dir):
    """
    Accuire the train, test, and the valid folders from the user provided data_dir.  It then sets up
    the data transform for both training and testing and uses it to get the image datasets for the data
    loaders.
    
    Parameters: 
     data_dir - String of the path of images to train, test, and validate the model.
    Returns:
     dataloaders - Contains the dataloader for training, testing, and validation.
     class_to_idx - class to index
    """
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {'train': transforms.Compose([transforms.RandomRotation(30),
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])]),
                       'test': transforms.Compose([transforms.Resize(255),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                                        [0.229, 0.224, 0.225])])}


    image_datasets = {'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                      'test': datasets.ImageFolder(test_dir, transform=data_transforms['test']),
                      'validation': datasets.ImageFolder(valid_dir, transform=data_transforms['test'])}


    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
                   'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64),
                   'validation': torch.utils.data.DataLoader(image_datasets['validation'], batch_size=64)}
    
    return dataloaders, image_datasets['train'].class_to_idx

def train_model(model, criterion, optimizer, epochs, device, dataloaders):
    """
    Trains the model with the training data and tests it with the testing data as it 
    prints out training loss, validation loss, and validation accuracy as the network trains
    Parameters:
     model - CNN model that will be trained
     criterion - The mathematical method for evaluating how well a model fits the data
     optimizer - The optimizer to use for training the model
     epochs - Number of cycles that the training dataset will be used to train the model
     device - The device GPU or CPU that will be used to train the model
     dataloader - Contaions the datasets for training and testing
    Returns:
     none
    """
    
    model.to(device)

    print("Start Training!")
    train_losses, test_losses = [], []
    with active_session():
        for epoch in range(epochs):
            running_loss = 0
            for inputs, labels in dataloaders['train']:

                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            else:
                test_loss = 0
                accuracy = 0

                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloaders['test']:
                        inputs, labels = inputs.to(device), labels.to(device)

                        logps = model(inputs)
                        test_loss += criterion(logps, labels)

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                train_losses.append(running_loss/len(dataloaders['train']))
                test_losses.append(test_loss/len(dataloaders['test']))

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/len(dataloaders['train']):.3f}.. "
                      f"Test loss: {test_loss/len(dataloaders['test']):.3f}.. "
                      f"Test accuracy: {accuracy/len(dataloaders['test']):.3f}")    
                model.train()

    print("Training Done!")

def save_model(model, optimizer, arch, hidden_layer, dropout, epochs, lr, class_to_idx, cp_path):
    """
    Adds the input parameters, optimizer, and model data to the checkpoint and saves the checkpoint
    tp the cp_path
    Parameters:
     model - The trained CNN model to save
     optimizer - The optimizer to use for training the model
     arch - The pre-trained model architecture that was used to create the model 
     hidden_layer - Number of hidden layers the model has
     dropout - The dropout used to train the model
     epochs - The number of training cycles
     lr - The learning rate that was used to train the model
     class_to_idx - class to index.
     cp_path - The path that the checkpoint will be saved to
    Returns:
     none 
    """
    
    print("Saveing checkpoint to [{}]".format(cp_path))
    model.cpu()
    checkpoint = {'network': arch,
                  'hl': hidden_layer,
                  'lr': lr,
                  'dropout': dropout,
                  'optimizer': optimizer.state_dict(),
                  'epochs': epochs,
                  'class_idx': class_to_idx,
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, cp_path)

def parse_args():
    """
    Retrieves and parses the 8 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 8 command line arguments. If 
    the user fails to provide some or all of the 8 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. Image Folder as --dawhich is required.
      2. CNN Model Architecture as --arch with default value 'vgg16'
      3. Save folder as --save_dir with the default value './checkpoint.pth'
      4. Learning Rate as --lr with the default value of '0.003'
      5. Hidden Layer as --hidden_layer with the default value of '515'
      6. epochs as --epochs with the default value of '10'
      7. dropout as --dropout with the default value of '0.2'
      8. gpu as --gpu uses the GPU instead of the CPU
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', action="store", type=str, help='Path to the folder with the images')
    parser.add_argument('--save_dir',type=str, default='./checkpoint.pth',help='Path to save checkpoints')
    parser.add_argument('--arch',type=str, default='vgg16',help='Model architecture that will be used')
    parser.add_argument('--lr',type=float, default=0.003,help='Learning rate used for training')
    parser.add_argument('--hidden_layer',type=int, default=512, help='Size of the hidden layer in traing model')
    parser.add_argument('--epochs',type=int, default=10, help='Amount of epochs')
    parser.add_argument('--dropout',type=float, default=0.2, help='Dropout rate used for training')
    parser.add_argument('--gpu', action='store_true', help='Use the GPU to train the model')
    
    return parser.parse_args()

#def main():
if __name__ == '__main__':

    args = parse_args()
    
    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    print("device: ", device)
    
    # Load the picture data
    data_loaders, class_to_idx = load_data(args.data_dir)
    # Create the model based on the input arguments
    model, criterion, optimizer = create_model(args.arch, args.dropout, args.hidden_layer, args.lr)
    # Train the model
    train_model(model, criterion, optimizer, args.epochs, device, data_loaders)
    # Validation of the model
    validate_model(model, device, data_loaders['validation'])
    # Save the model
    save_model(model, optimizer, args.arch, args.hidden_layer, args.dropout, args.epochs, args.lr, class_to_idx, args.save_dir)