# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 15:21:04 2023

@author: xiaoke.mu

Integrating DPC data
"""
import numpy as np
from numpy import pi
#from numpy.fft import fft2, ifft2
from scipy.fft import fft2, ifft2, fftfreq
import matplotlib.pyplot as plt


def zero_pad(img, new_shape):
    """
    Zero-pads an image to a new shape

    Parameters:
    img (2D numpy array): Input image
    new_shape (tuple of ints): New shape for the image (height, width)

    Returns:
    2D numpy array: Zero-padded image with shape `new_shape`
    """
    height, width = img.shape
    pad_height, pad_width = new_shape[0] - height, new_shape[1] - width
    pad_top, pad_bottom = pad_height//2, pad_height//2 + pad_height%2
    pad_left, pad_right = pad_width//2, pad_width//2 + pad_width%2

    padded_img = np.zeros(new_shape, dtype=img.dtype)
    padded_img[pad_top:-pad_bottom, pad_left:-pad_right] = img

    return padded_img

def int_2d_fourier(CenterX,CenterY, sampling):
    #CenterX = zero_pad(CenterX, np.array(CenterX.shape)+100)
    #CenterY = zero_pad(CenterY, np.array(CenterY.shape)+100)
    #from scipy.ndimage import zoom
    #CenterX = zoom(CenterX,2)
    #CenterY = zoom(CenterY,2)
    #x = np.arange(0,CenterX.shape[1]*sampling,sampling)
    #y = np.arange(0,CenterX.shape[0]*sampling,sampling)
    #r = np.meshgrid(x,y)
    
    k_x = fftfreq(CenterX.shape[1], sampling)
    k_y = fftfreq(CenterX.shape[0], sampling)
    # full coordinate arrays
    k_xx, k_yy = np.meshgrid(k_x, k_y)
    k_sq = np.where(k_xx**2+k_yy**2 != 0, k_xx**2+k_yy**2, (k_x[1]-k_x[0])/10000)
    
    iDPC = -ifft2((fft2(CenterX) * k_xx + fft2(CenterY) * k_yy) / (2*np.pi * 1j * k_sq))
    
   # from PIL import Image
   # re = Image.fromarray(iDPC.real)
   # re_binned = np.array(re.resize((re.size[0]//2, re.size[1]//2), Image.LANCZOS))
   # Imag = Image.fromarray(iDPC.imag)
   # Imag_binned = np.array(Imag.resize((Imag.size[0]//2, Imag.size[1]//2), Image.LANCZOS))     
    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(20,20))
    # Plot first image
    axs[0,0].imshow(CenterX,vmin=-3.3, vmax=3.3)
    axs[0,0].set_title('X')
    axs[0,1].imshow(CenterY,vmin=-3.3, vmax=3.3)
    axs[0,1].set_title('Y')
    axs[1,0].imshow(-np.real(iDPC),vmin=-25, vmax=15)
    #axs[1,0].imshow(re_binned,vmin=-0.03, vmax=0.04)
    axs[1,0].set_title('Real')
    axs[1,1].imshow(np.imag(iDPC))
    #axs[1,1].imshow(Imag_binned)
    axs[1,1].set_title('Imaginary')
    for ax in axs.flat:
        ax.set_axis_off()
    plt.show()
    
    return -1*np.real(iDPC)

def divergence(x, y, u, v):
    return np.gradient(u)[1]/np.gradient(x) + np.gradient(v)[0]/np.gradient(y)


def int_2d_vonPoisonEquition(CenterX,CenterY, sampling):
    x = np.arange(0,CenterX.shape[1]*sampling,sampling)
    y = np.arange(0,CenterX.shape[0]*sampling,sampling)
    #r = np.meshgrid(x,y)
    k_x = fftfreq(CenterX.shape[1], sampling)
    k_y = fftfreq(CenterX.shape[0], sampling)
    k = np.meshgrid(k_x,k_y)
    k_squ = k[0]**2+k[1]**2
    k_squ = np.where(k_squ != 0, k_squ, (k_x[1]-k_x[0])/1000)
    charge = divergence(x, y, CenterX, CenterY)
    iDPC = ifft2(fft2(charge)/(-4*pi*pi*k_squ))
    
    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(20,20))
    # Plot first image
    axs[0,0].imshow(CenterX,vmin=-0.3, vmax=0.3)
    axs[0,0].set_title('X')
    axs[0,1].imshow(CenterY,vmin=-0.3, vmax=0.3)
    axs[0,1].set_title('Y')
    axs[1,0].imshow(np.real(iDPC),vmin=-3, vmax=3)
    axs[1,0].set_title('Real')
    axs[1,1].imshow(np.imag(iDPC))
    axs[1,1].set_title('Imaginary')
    return np.real(iDPC)