#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 14:13:18 2018

@author: iswariya
"""
import torch
import torch.optim as optim
from metric import Metric
import copy



def unfreeze_layers(model):
    '''Function to unfreeze the layers of a model
    
    Parameters
    ----------
    model : object
        Object of class UNet which has the network
        
    Returns
    -------
    List
        List of model parameters
        
    '''
    
    for param in model.parameters():
        param.requires_grad = True
    
    params = list(model.parameters())
    
    return params


def fit(model, criterion, optimizer, **kwargs):
    '''Function to train the model
    
    Parameters
    ----------
    model : object
        Object of class UNet which has the network
    criterion : pytorch loss module
        Contains the loss function for optimizing the model
    optimizer : pytorch optimizer
        Contains the desired optimizer to optimize the model parameters
    **kwargs : dict
        Dictionary of keywords passed to the function
        
    Returns
    -------
    float
        Loss value per epoch
    List of floats
        Loss per batch per epoch
    tuple
        List of iou per class and mean iou
    '''
    
    
    if kwargs['phase'] == 'Training':
        model.train()
    if kwargs['phase'] == 'Validation':
        model.eval()
    
    device = kwargs['device']
    running_loss = 0
    batch_wise_loss = []
    metric_val = Metric(4)
    metric_val.reset()

    
    for i, data in enumerate(kwargs['dataloader']):
        
        images = data['image'].to(device)
        masks = data['mask'].to(device)
        
        output = model(images)
        
        loss = criterion(output, masks)
        
        running_loss += loss.item()
        
        batch_wise_loss.append(loss.item())
        
        # Varying learning rate if lr_finder is used
        if kwargs['lr_finder'] == True:
            
            optimizer = optim.Adam(kwargs['parameters'], lr = kwargs['lr_range_val'][i])

        if kwargs['phase'] == 'Training':
            #Backward
            optimizer.zero_grad()
            loss.backward()

            #Update weights
            optimizer.step()
        
        pred = torch.argmax(output, dim=1)
        iou = metric_val.get_iou(pred, masks)

        
    return running_loss/len(kwargs['dataloader']), batch_wise_loss, iou


def train(train_loader_low_resolution, train_loader_full_resolution, val_loader_low_resolution, val_loader_full_resolution, model, criterion, params, cuda0):
    
    train_loss_values = []
    val_loss_values = []
    mean_iou = 0
    unfreeze_flag = True
    optim_flag = True
    learning_rate = 0.001  # Optimal learning rate using lr finder
    
    optimizer = optim.Adam(params, lr = learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience =1)
    
    for epoch in range(80):
        
        if epoch>25 and epoch<=40:
            
            if unfreeze_flag:
                params = unfreeze_layers(model)
                optimizer = optim.Adam(params, lr = 0.0001)
                unfreeze_flag = False
            
            train_loss, _, train_iou = fit(model, criterion, optimizer, parameters=params, dataloader=train_loader_low_resolution, phase='Training', lr_range_val=[], lr_finder=False, device=cuda0)
            train_loss_values.append(train_loss)
            val_loss, _, val_iou = fit(model, criterion, optimizer, parameters=params, dataloader=val_loader_low_resolution, 
                                       phase='Validation', lr_range_val=[], lr_finder=False, device=cuda0)
            val_loss_values.append(val_loss) 
            
        elif epoch>40:
            
            if optim_flag:
                optimizer = optim.Adam(params, lr = 0.00001)
                optim_flag = False
            
            train_loss, _, train_iou = fit(model, criterion, optimizer, parameters=params, dataloader=train_loader_full_resolution, phase='Training', lr_range_val=[], lr_finder=False, device=cuda0)
            train_loss_values.append(train_loss)
            val_loss, _, val_iou = fit(model, criterion, optimizer, parameters=params, dataloader=val_loader_full_resolution, 
                                       phase='Validation', lr_range_val=[], lr_finder=False, device=cuda0)
            val_loss_values.append(val_loss)
            
        else:
            
            train_loss, _, train_iou = fit(model, criterion, optimizer, parameters=params, dataloader=train_loader_low_resolution, phase='Training', lr_range_val=[], lr_finder=False, device=cuda0)
            train_loss_values.append(train_loss) 
            
            val_loss, _, val_iou = fit(model, criterion, optimizer, parameters=params, dataloader=val_loader_low_resolution, 
                                       phase='Validation', lr_range_val=[], lr_finder=False, device=cuda0)
            val_loss_values.append(val_loss)
        
        
        print(f'Epoch: {epoch}   Train Loss: {train_loss:.5f}  IoU: {train_iou[0]}  Mean IoU:{train_iou[1]:.5f}')
        print(f'Epoch: {epoch}   Val Loss: {val_loss:.5f}      IoU: {val_iou[0]}    Mean IoU:{val_iou[1]:.5f}')
    
        if val_iou[1]>mean_iou:
        
            model_checkpoint = copy.deepcopy(model.state_dict())
    
        scheduler.step(val_loss)
    
    return model_checkpoint

    
