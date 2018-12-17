#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 14:14:39 2018

@author: iswariya
"""
import torch
import torch.nn as nn


class UNet(nn.Module):
    
    '''This class implements an encoder-decoder similar to UNet'''
    
    
    def __init__(self, resnet_layers):
        '''Constructor method used to create the encoder layers, decode layers and lateral connections'''
        
        super(UNet, self).__init__()
        
        self._encoder_layer_1 = nn.Sequential(*list(resnet_layers.children())[:5])
        
        self._encoder_layer_2 = nn.Sequential(*list(resnet_layers.children())[5])
        
        self._encoder_layer_3 = nn.Sequential(*list(resnet_layers.children())[6])
        
        self._encoder_layer_4 = nn.Sequential(*list(resnet_layers.children())[7])
        
        
        self._decoder_layer_1 = nn.Sequential(nn.ReLU(),
                                      nn.ConvTranspose2d(512, 128, 3, 2, 1, 1))
        
        self._decoder_layer_2 = nn.Sequential(nn.BatchNorm2d(256),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(256, 128, 3, 2, 1, 1))
        
        self._decoder_layer_3 = nn.Sequential(nn.BatchNorm2d(256),
                              nn.ReLU(),
                              nn.ConvTranspose2d(256, 128, 3, 2, 1, 1))
        
        self._decoder_layer_4 = nn.Sequential(nn.BatchNorm2d(256),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(256, 4, 3, 1, 1))
        
        self._lateral_conv_1 = nn.Conv2d(64, 128, 1, 1)
        self._lateral_conv_2 = nn.Conv2d(128, 128, 1, 1)
        self._lateral_conv_3 = nn.Conv2d(256, 128, 1, 1)
        
    def encoder(self, x):
        '''Class method used to group all encoder layers
        
        Parameters
        ----------
        x : Float tensor
            Input image tensor to the model of size BATCH_SIZE x C x H x W.
            C - 3
            H, W - 512 for low resolution images and 1024 for full resolution images
            
        Returns
        -------
        Float tensor
            Returns a downsampled feature map of size BATCH_SIZE x 512 x H x W.
            
        '''
        
        x = self._encoder_layer_1(x)
        self._lateral_outputs.append(x)
        x = self._encoder_layer_2(x)
        self._lateral_outputs.append(x)
        x = self._encoder_layer_3(x)
        self._lateral_outputs.append(x)
        x = self._encoder_layer_4(x)
        
        return x
    
    def decoder(self, x):
        '''Class method used to group all decoder layers
        
        Parameters
        ----------
        x : Float tensor
            Output tensor from encoder of size BATCH_SIZE x 512 x H x W
            
        Returns
        -------
        Float tensor
            Returns an upsampled feature map of size BATCH_SIZE x C x H/4 x W/4.
            C - No. of classes of objects in the image. (4 classes)
            Here we have 3 classes (Ball, field, field lines) + 1 background class.
            The decoder outputs an image whose height and width is reduced 4 times.
            
        '''
        
        x = self._decoder_layer_1(x)
        lateral_output_1 = self._lateral_conv_3(self._lateral_outputs[-1])
        x = torch.cat((x, lateral_output_1), dim=1)
        
        x = self._decoder_layer_2(x)
        lateral_output_2 = self._lateral_conv_2(self._lateral_outputs[-2])
        x = torch.cat((x, lateral_output_2), dim=1)
        
        x = self._decoder_layer_3(x)
        lateral_output_3 = self._lateral_conv_1(self._lateral_outputs[-3])
        x = torch.cat((x, lateral_output_3), dim=1)
        
        x = self._decoder_layer_4(x)
        
        return x
        
    def forward(self, x):
        '''Class method for forward pass through the model.
        
        Parameters
        ----------
        x : Float tensor
            Input image tensor to the model of size BATCH_SIZE x C x H x W.
            C - 3
            H, W - 512 for low resolution images and 1024 for full resolution images
            
        Returns
        -------
        Float tensor
            Returns an upsampled feature map of size BATCH_SIZE x C x H/4 x W/4.
            C - No. of classes of objects in the image. (4 classes)
            Here we have 3 classes (Ball, field, field lines) + 1 background class.
            The decoder outputs an image whose height and width is reduced 4 times.
        
        '''
        
        self._lateral_outputs = []
        
        x = self.encoder(x)
        x = self.decoder(x)

        return x