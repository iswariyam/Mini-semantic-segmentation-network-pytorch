#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 14:13:47 2018

@author: iswariya
"""
from metric import Metric
import numpy as np
import torch


def predict(model, testloader, criterion, **kwargs):
    '''Function to evaluate the model
    
    Parameters
    ----------
    model : object
        Object of class UNet which has the network
    testloader : pytorch dataloader object
        iter object for loading data from test set
    criterion : pytorch loss module
        Contains the loss function for optimizing the model
        
    Returns
    -------
    List
        List of predicted mask tensors output from the model
    List
        List of annotated mask tensors from the dataset
    Tuple
        List of iou per class and mean iou
        
    '''
    
    model.eval()
    metric_val = Metric(4)
    metric_val.reset()
    running_loss = 0
    
    device = kwargs['device']
    batch_size = kwargs['batchsize']
    prediction_array = np.ndarray((batch_size, 120, 160))
    mask_array = np.ndarray((batch_size, 120, 160))

    
    for i, data in enumerate(testloader):
        
        images = data['image'].to(device)
        masks = data['mask'].to(device)
        
        output = model(images)
        
        loss = criterion(output, masks)
        
        running_loss += loss.item()
        
        pred = torch.argmax(output, dim=1)
        iou = metric_val.get_iou(pred, masks)
        
        prediction_array = np.vstack((prediction_array, pred.detach().cpu().numpy()))
        #print(masks.detach().cpu().numpy().shape, mask_array.shape)
        mask_array = np.vstack((mask_array, masks.detach().cpu().numpy()))
        #print(prediction_array.shape, mask_array.shape)

    return prediction_array, mask_array, iou, metric_val.confusion_matrix