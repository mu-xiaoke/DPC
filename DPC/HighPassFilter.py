# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 12:40:11 2023

@author: xiaoke.mu
"""
import numpy as np
from numpy.fft import fft2, ifft2, fftfreq, fftshift
import matplotlib.pyplot as plt
def high_pass_2d(img, strength_percent):
    radius = np.round(strength_percent*img.shape[0]/2).astype(int)
    y = np.arange(img.shape[0])
    x = np.arange(img.shape[1])
    k_x = fftfreq(img.shape[1], 1)
    k_y = fftfreq(img.shape[0], 1)
    k_xx, k_yy = np.meshgrid(k_x, k_y)
    mask = np.sqrt(k_xx**2+k_yy**2)
    mask = np.where(mask < radius*(k_x[1]-k_x[0]), mask/mask[0,radius], 1)
    filtered_img = np.real(ifft2(fft2(img)*mask))
    
    # # Plot result
    # fig, axs = plt.subplots(1, 3, figsize=(20,10))
    # axs[0].imshow(img,vmin=-16,vmax=26)
    # axs[0].set_title('Original')
    # axs[1].imshow(fftshift(mask))
    # axs[1].set_title('Mask')
    # #axs[2].imshow(filtered_img,vmin=np.min(filtered_img[64:192,64:192]),vmax=np.max(filtered_img[64:192,64:192]))
    # axs[2].imshow(filtered_img, vmin = -3, vmax=3)
    # axs[2].set_title('Filtered')
    
    return filtered_img