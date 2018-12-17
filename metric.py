#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 14:34:43 2018

@author: iswariya
"""
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

class Metric():
    
    '''Class for computing different metrics like IoU'''
    
    
    def __init__(self, num_classes):
        '''Constructor method which initializes the confusion matrix
        
        Parameters
        ----------
        num_classes : int
            No. of classes of objects present in the input image
            
        '''
        
        self.confusion_matrix = np.ndarray((num_classes, num_classes), dtype=np.int64)
        
    def reset(self):
        '''Method to reset the confusion matrix.
        
        It is usually called after one epoch.
        '''
        
        self.confusion_matrix.fill(0)
        
    def get_confusion_matrix(self, predicted, target):
        '''Method to generate a confusion matrix using the predicted and target images.
        
        Parameters
        ----------
        predicted : Float tensor
            A batch of masks output from the model
        target : Float tensor
            A batch of corresponding annotated masks from the dataset
            
        '''
        
        predicted_val = predicted.cpu().detach().numpy()
        target_val = target.detach().cpu().detach().numpy()
        cm = confusion_matrix(target_val.flatten(), predicted_val.flatten())
        self.confusion_matrix += cm
        
    def get_iou(self, predicted, target):
        '''Method to compute the IoU from the confusion matrix
        
        Parameters
        ----------
        predicted : Float tensor
            A batch of masks output from the model
        target : Float tensor
            A batch of corresponding annotated masks from the dataset
        
        '''
        
        self.get_confusion_matrix(predicted, target)
        true_positive = np.diag(self.confusion_matrix)
        false_positive = np.sum(self.confusion_matrix, axis=0) - true_positive
        false_negative = np.sum(self.confusion_matrix, axis=1) - true_positive
        iou = true_positive/(true_positive + false_positive + false_negative)
        
        return (iou, np.nanmean(iou))
    

def plot_confusion_matrix(cm, classes):
    '''Function to plot the confusion matrix
    
    Parameters
    ----------
    cm
        Numpy array of confusion matrix
    classes
        List of number of classes of object
    
    '''
    
    tick_marks = np.arange(len(classes))
    cmap = plt.get_cmap('Greens')
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.set(xlabel="Predicted Values", ylabel="True Values", xticks=tick_marks, yticks=tick_marks, 
           xticklabels=classes, yticklabels=classes)
    img = ax.imshow(cm, cmap = cmap, interpolation = 'nearest')
    fig.colorbar(img, ax=ax)
    ax.set_aspect('auto')

    print(classes)
    
    threshold = cm.max()/2
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        
        ax.text(i, j, f'{cm[i, j]}' , horizontalalignment = 'center', 
                 color="white" if cm[i, j] > threshold else "black")

    plt.savefig("Results/Confusion_matrix.png")
    plt.close()