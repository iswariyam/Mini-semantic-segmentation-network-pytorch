#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 14:33:25 2018

@author: iswariya
"""
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model import UNet
from dataloader import SoccerDataset, get_dataloader
from metric import plot_confusion_matrix
from utils import load_sample_image, CyclicLearningRate, lr_finder_plot, plot_mask
from train import fit, train
from evaluate import predict




if __name__ == '__main__':
    
    cuda0 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(cuda0)
    print(torch.cuda.is_available())
    print("yes")
    
    PATH = '/opt/datasets/Soccer_dataset/Train_1/'
    image_folder = 'JPEG_images'
    mask_folder = 'Annotations'
    BATCH_SIZE = 8
    full_resolution = (640, 480)
    downsampled_resolution = (320, 160)
    
    soccer_dataset_low_resolution = SoccerDataset(PATH, image_folder, mask_folder, 
                                                  downsampled_resolution, transform=['Horizontal_Flip', 'Brightness_adjust'])
    soccer_dataset_full_resolution = SoccerDataset(PATH, image_folder, mask_folder, 
                                                   full_resolution, transform=['Horizontal_Flip', 'Brightness_adjust'])
    
    train_loader_low_resolution, val_loader_low_resolution = get_dataloader(soccer_dataset_low_resolution, BATCH_SIZE)
    train_loader_full_resolution, val_loader_full_resolution = get_dataloader(soccer_dataset_full_resolution, BATCH_SIZE)

    load_sample_image(train_loader_low_resolution)
    
    resnet_layers = models.resnet18(pretrained=True)
    
    for param in resnet_layers.parameters():
        param.requires_grad = False
        
    model = UNet(resnet_layers)
    model = nn.DataParallel(model)
    model = model.to(cuda0)
       
    params = list(model.parameters())[61:]
    
    learning_rate = 0.001
    optimizer = optim.Adam(params, lr = learning_rate)
    criterion = nn.CrossEntropyLoss().to(cuda0)

    learning_rate = CyclicLearningRate(train_loader_low_resolution, 0.1, 1e-5)
    learning_rate_array = learning_rate.get_learning_rate()

    _, batch_loss, _ = fit(model, criterion, optimizer, parameters=params, dataloader=train_loader_low_resolution, 
                       phase='Training', lr_range_val=learning_rate_array, lr_finder=True, device=cuda0)
    lr_finder_plot(learning_rate_array, batch_loss)

    model_checkpoint = train(train_loader_low_resolution, train_loader_full_resolution, val_loader_low_resolution, val_loader_full_resolution, model, criterion, params, cuda0)
    model.load_state_dict(model_checkpoint)
    torch.save(model.state_dict(), "Unet.th")
    
    prediction, label, iou, cm = predict(model, val_loader_full_resolution, criterion, device=cuda0, batchsize=BATCH_SIZE)
    cmap = np.array([[0, 0, 0], [245, 130, 48], [0, 130, 200], [60, 180, 75]], dtype=np.uint8)

    x = np.array(prediction[9], dtype=np.uint8)
    plot_mask(prediction[9:15], label[9:15], cmap)
    classes = ('Backround', 'Ball', 'Field Lines', 'Field')
    plot_confusion_matrix(cm.T, classes)