# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 15:02:26 2023

@author: xiaoke.mu
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_field(field):
    # Normalize the field magnitude
    field = field / np.abs(field).max()

    # Compute the hue from the orientation angle
    hue = np.angle(field) / np.pi

    # Convert to an RGB image
    rgb = plt.get_cmap('hsv')(hue)

    # Plot the image
    plt.imshow(rgb, extent=(0, field.shape[1], 0, field.shape[0]), aspect='equal')
    plt.axis('off')
    plt.show()

# Example usage
field = np.exp(1j * np.linspace(0, np.pi, 100))
field = np.tile(field, (100, 1))
plot_field(field)