#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 14:14:54 2018

@author: iswariya
"""
import numpy as np
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class SoccerDataset(Dataset):
    
    '''This is a class for creating an iterable dataset of images and annotated masks'''
    
    
    def __init__(self, path, image_dir, mask_dir, image_size, transform=None):
        '''Constructor method for the class
            
            Parameters
            ----------
            path : string
                Path of the dataset directory.
            image_dir : string
                Name of folder where images are stored
            mask_dir : string
                Name of folder where annotated masks are stored
            image_size : int
                Size to which the input image has to be resized.
            transform : string
                Name of transforms to be applied to the dataset. Eg: h_flip (horizontal flip)    
                
        '''
        
        super(SoccerDataset, self).__init__()
        
        self._path = path
        self._image_dir = image_dir
        self._mask_dir = mask_dir
        self._transform = transform
        self._image_size = image_size
        self._file_list = os.listdir(self._path+image_dir)
        
    def __len__(self):
        '''This function returns the number of images in the dataset'''
        
        return len(self._file_list)
        
    def __getitem__(self, idx):
        '''This function returns a tensor of an image and its mask
        
        The function iterates through the images as per the index argument, reads them and
        resizes them according to the required image size, and returns a dictionary of
        image and annotated mask tensors.
        
        Parameters
        ----------
        idx : int
            Index of the image to be read
            
        Returns
        -------
        dict 
            An image - mask tensor pair is returned as a dictionary
            
        '''
        
        image_name = self._file_list[idx]
        image_path = os.path.join(self._path, self._image_dir, image_name)
        mask_path = os.path.join(self._path, self._mask_dir, image_name.split('.')[0]+ '.png')
        
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, 0)
        
        image = cv2.resize(image, (self._image_size[0], self._image_size[1]))
        mask = cv2.resize(mask, (self._image_size[0]//4, self._image_size[1]//4))
        
        image, mask = self._get_transforms(image, mask, self._transform)

        image = np.transpose(image, [2,0,1]).astype(np.float32)

        img_mask_pair = {'image':torch.from_numpy(image), 'mask':torch.LongTensor(mask)}
        
        #print(img_mask_pair['image'].size(), img_mask_pair['mask'].size())
        
        return img_mask_pair
    
    def _get_transforms(self, image, mask, transforms):
        '''Method to implement transforms to be applied on dataset
        
        Only horizontal flip transform which is required for this project is implemented.
        Additional transforms can be implemented by added several if else loops in the 
        code below.
        
        Parameters
        ----------
        tranforms : List
            List of string names of transforms.
            
        Returns
        -------
        Numpy array
            Numpy array of transformed image is returned
        Numpy array
            Numpy array of transformed mask is returned
            
        '''
        
        for i in transforms:
            
            if i=='Horizontal_Flip':
                
                p = np.random.rand()
                
                if p>0.5:
                    image = cv2.flip(image, 1)
                    mask = cv2.flip(mask, 1)
                    
            if i=='Brightness_adjust':
                
                p = np.random.rand()
                
                if p>0.5:
                    
                    image = self.brightness_augment(image)
                    
                
        return image, mask
    
    

    def brightness_augment(self, img, factor=0.5): 
        
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:, :, 2] = hsv[:, :, 2] * (factor + np.random.uniform()) 
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255 
        rgb = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2RGB)
        
        return rgb
    
    
def get_dataloader(dataset, batchsize):
    '''This function returns the train and validation dataloader.
    
    It implements a train-validation split using SubsetRandomSampler in Pytorch.
    We use a train-val split of 80-20%. First random 80% of the images in the dataset
    are chosen and fed into the train_sampler. The remaining 20% are then put into val_sampler.
    
    Parameters
    ----------
    dataset : object of Dataset class
        Object used to get image, label tensor pair from the dataset folder
    
    Returns
    -------
    list of iter objects
        Returns dataset iter object for training and validation set
        
    '''
    
    train_set_size = int(0.8 * dataset.__len__())   
    train_indices = np.random.choice(np.arange(dataset.__len__()), train_set_size, replace = False) 
    train_sampler = SubsetRandomSampler(train_indices)

    val_indices = np.setdiff1d(np.arange(dataset.__len__()), train_indices, assume_unique= True)
    val_sampler = SubsetRandomSampler(val_indices)

    trainloader = DataLoader(dataset, batch_size = batchsize, sampler=train_sampler, num_workers=2)
    valloader = DataLoader(dataset, batch_size = batchsize, sampler=val_sampler, num_workers=2)
    
    return trainloader, valloader
