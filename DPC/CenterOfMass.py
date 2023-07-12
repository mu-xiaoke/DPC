# -*- coding: utf-8 -*-
"""
Center of Mass and iDPC

Created on Mon Jan 30 11:11:13 2023

@author: xiaoke.mu
"""

import numpy as np

def center(Img):
    one_x = np.ones(Img.shape[1])
    one_y = np.ones(Img.shape[0])
    n_x = np.arange(0,Img.shape[1],1)
    n_y = np.arange(0,Img.shape[0],1)
   
    a = np.array([n_y,one_y])
    b = np.transpose([one_x,n_x])
    #Img[Img<0] = 0 #set all negative value to zero
    Img = np.clip(Img, a_min=0, a_max=None) #set all negative value to zero
    c = a@Img@b # @ means Matrix multiplication
    
    momentX =  c[1,1] / c[1,0] # column
    momentY =  c[0,0] / c[1,0] # row
    return [momentX, momentY]

def CenterMap(data_4D):
    ## ref = an array of the reference center
    Data_shape = data_4D.shape
    CenterX = np.zeros([Data_shape[0],Data_shape[1]])
    CenterY = np.zeros([Data_shape[0],Data_shape[1]])
    for i in range(Data_shape[0]):
        for j in range(Data_shape[1]):
            CenterX[i,j], CenterY[i,j] = center(data_4D[i,j])
    #Correct the center x and y using their average
    CenterX = CenterX-np.average(CenterX)
    CenterY = CenterY-np.average(CenterY)
    return [CenterX, CenterY]