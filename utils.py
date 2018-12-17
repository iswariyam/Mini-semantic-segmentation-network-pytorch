#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 14:13:58 2018

@author: iswariya
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

        
class CyclicLearningRate:
    '''Implementation based on Smith, Leslie N. "Cyclical learning rates for training neural networks."'''
    
    def __init__(self, dataloader, max_lr, base_lr):
        '''Constructor method to initialize parameters of learning rate generator
        
        Parameters
        ----------
        dataloader : iter object
            Iter object which iterates through the dataset of images
        base_lr : float
            Initial learning rate
        max_lr : float
            Maximum learning rate
        
        '''
        
        self.num_iterations = len(dataloader)  
        self.stepsize = len(dataloader)
        self.max_lr = max_lr     
        self.base_lr = base_lr
        
    def calculate_triangular_learning_rate(self, iteration):
        '''Method to generate triangular learning rates 
        
        Parameters
        ----------
        iteration : int
            Iteration number

        Returns
        -------
        Float
            Returns the learning rate for each iteration
        '''

        cycle = np.floor(1 + iteration/(2  * self.stepsize))
        x = np.abs(iteration/self.stepsize - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1-x))

        return lr
    
    def get_learning_rate(self):
        '''Method to get the learning rate for each iteration
        
        Returns
        -------
        Float list
            List of learning rates for one epoch
        
        '''
        
        lr_array = []
        for iteration in range(self.num_iterations):
    
            lr = self.calculate_triangular_learning_rate(iteration)
            lr_array.append(lr)
            
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set(title = 'LR plot', xlabel = 'iterations', ylabel = 'learning rate', yscale = 'log')
        ax.plot(lr_array)
        plt.savefig("Results/learning_rate_plot.png")
        plt.close()
        
        return lr_array


def plot_mask(prediction, label, cmap):
    '''Function to plot the annotated masks and the predicted masks
    
    Parameters
    ----------
    prediction
        List of predicted mask tensors output from the model
    label 
        List of annotated mask tensors from the dataset
    cmap
        Numpy array of confusion matrix
         
    '''
    
    path = os.getcwd()
    image_folder = '/Results/'
    image_path = path + image_folder
    
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    
    
    for i in range(1, prediction.shape[0]):
        
        label_mask = cmap[np.array(label[i], dtype=np.uint8)]
        
        fig = plt.figure(figsize=(8,10))
        ax = fig.add_subplot(prediction.shape[0], 1, i)
        ax.set(title='True Mask')
        ax.set_axis_off()
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.imshow(cv2.cvtColor(label_mask, cv2.COLOR_BGR2RGB))
        plt.savefig(os.path.join(image_path, 'label_{0}.png'.format(i)), bbox_inches='tight', pad_inches=0)
    
    for i in range(1, prediction.shape[0]):
        
        prediction_mask = cmap[np.array(prediction[i], dtype=np.uint8)]
        
        fig1 = plt.figure(figsize=(6, 8))
        ax1 = fig1.add_subplot(prediction.shape[0], 1, i)
        ax1.set(title='Predicted Mask')
        ax1.set_axis_off()
        ax1.get_yaxis().set_visible(False)
        ax1.get_xaxis().set_visible(False)
        ax1.imshow(cv2.cvtColor(prediction_mask, cv2.COLOR_BGR2RGB))
        plt.savefig(os.path.join(image_path, 'predicted_mask_{0}.png'.format(i)), bbox_inches='tight', pad_inches=0)
        plt.tight_layout()
        
        
def lr_finder_plot(lr_array, batch_loss):
    '''Function to plot learning rate vs batch loss
    
    Parameters
    ----------
    lr_array : list
        List of learning rates per iteration
    batch_loss : list
        List of losses per iteration
        
    '''
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set(title='lr finder', xlabel='learning rate', ylabel='loss', xscale='log')
    ax1.plot(lr_array, batch_loss)
    plt.savefig("Results/lr_finder.png")
    plt.close()
    
    
def load_sample_image(dataloader):
    '''Function to load and display a sample image from the dataset
    
    Parameters
    ----------
    dataloader : iter object
        Iter object which iterates through the dataset of images
    
    '''
    
    obj = iter(dataloader)
    data = obj.next()
    
    image = data['image'][0].numpy()
    image = np.transpose(image, [1, 2, 0])
    image = cv2.resize(image, (320, 160))
    image = image.astype(int)
    plt.imshow(image)

    plt.savefig("Results/Sample_image")
    plt.show()
 
    mask = data['mask'][0].numpy()
    print(mask.shape)

    plt.imshow(mask)
    plt.savefig("Results/Sample_mask")
    return image, mask
    