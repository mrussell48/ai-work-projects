#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */ImageClassifier/predict.py
#                                                                             
# PROGRAMMER: Michael J Russell Jr
# DATE CREATED: 4/20/2021
# REVISED DATE: 4/24/2021
# PURPOSE: Uses a loaded trained CNN model to predict the class and probabilities of 
#           an imge.
import argparse

import torch
import json

from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from model import create_model

def parse_args():
    """
    Retrieves and parses the 5 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 5 command line arguments. If 
    the user fails to provide some or all of the 5 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. Image as image which is required.
      2. Checkpoint as checkpoint which is required
      3. Top K as --top_k with the default value of '5'
      4. category_names as --category_names with no default value
      5. gpu as --gpu uses the GPU instead of the CPU
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    
    parser = argparse.ArgumentParser()

    parser.add_argument('image', action="store", type=str, help='Image to to predict.')
    parser.add_argument('checkpoint', action="store", type=str, help='Path to the checkpoint save file.')
    
    parser.add_argument('--top_k', type=int, default=5, help='Number of Top most likely cases to return.')
    parser.add_argument('--category_names', type=str, help='Mapping of categories to real names.')
    parser.add_argument('--gpu', action='store_true', help='Use the GPU to train the model.')
    
    return parser.parse_args()

def load_model(cp_path, device):
    """
    Loads an image as a PIL and transform it by scaleing, croping, and normalizing
    the image so it can run through the trained PyTorch model
    Parameters:
     cp_path - Path of the checkpoint pth file to load into a model
     device - The device to use
    Returns:
     model - The loaded trained model
     criterion - The loaded criterion
     optimizer - The loaded optimizer
    """
    print("Loading checkpoint to [{}]".format(cp_path))
    
    checkpoint = torch.load(cp_path, map_location=lambda storage, loc: storage)
    model, criterion, optimizer = create_model(checkpoint['network'], checkpoint['dropout'], checkpoint['hl'],checkpoint['lr'])

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_idx']
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    return model, criterion, optimizer

def process_image(image):
    """
    Loads an image as a PIL and transform it by scaleing, croping, and normalizing
    the image so it can run through the trained PyTorch model
    Parameters:
     image - Image to process 
    Returns:
     top_prob_array - The topk predicted  probabilities of the image
     top_classes - The topk predicted classes of the imager
    """

    pil_img = Image.open(image)
    pil_img.load()
    
    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = transform(pil_img)

    return img

def predict(image_path, model, device, topk):
    """
    Uses the trained model to predict what class and with what probability the image is of the class.
    It will return the topk classes and propabilities.
    Parameters:
     image_path - The image to predict 
     model - The loaded trained model that will be used to predict the image class
     device - The device that will be used to predict
     topk - The top k number of class and props
    Returns:
     top_prob_array - The topk predicted  probabilities of the image
     top_classes - The topk predicted classes of the imager
    """

    model.to(device)
    model.eval()
    
    img = process_image(image_path)
    img = img.unsqueeze(0)

    with torch.no_grad():
        output = model.forward(Variable(img).to(device))
        top_prob, top_labels = torch.topk(output, topk)
        top_prob = top_prob.exp()
        top_prob_array = top_prob.cpu().data.numpy()[0]
    
        inv_class_to_idx = {v: k for k, v in model.class_to_idx.items()}
    
        top_labels_data = top_labels.cpu().data.numpy()
        top_labels_list = top_labels_data[0].tolist()  
    
        top_classes = [inv_class_to_idx[x] for x in top_labels_list]
    
    return top_prob_array, top_classes

def load_json(json_pth):
    """
    Loads json file into a python dictionary
    Parameters:
     json_pth - The path to the category_names
    Returns:
     cat_to_name - Dictionar of the load category to names
    """
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def check(image_path, props, classes, name_map={}):
    """
    Prints out the top K classes and props.  If a category to names
    json was provided then it will print on the names instead of the
    classes.
    Parameters:
     image_path - Path to the image 
     props - The top K number of probabilities
     classes - The Top K number of classes
     name_map - Dictionary of the category
    Returns:
     none
    """
    true_class = image_path.split('/')[-2]
    if name_map:
        true_class = name_map[true_class]
    
    print('\nTrue Class of the Image: {}\n'.format(true_class))
    
    for cl, prop in zip(classes, props):
        if name_map:
            print('{:20}: {:.4f}'.format(name_map[cl], prop))
        else:
            print('{:5}: {:.4f}'.format(cl, prop))  
    
if __name__ == '__main__':
    args = parse_args()
    
    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    print("device: ", device)
    
    model, criterion, optimizer = load_model(args.checkpoint, device)
    props, classes = predict(args.image, model, device, args.top_k)
    
    if args.category_names:
        check(args.image, props, classes, load_json(args.category_names))
    else:
        check(args.image, props, classes)