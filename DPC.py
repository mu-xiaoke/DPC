# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 09:25:29 2023

@author: py4dstem
"""

import py4DSTEM
import matplotlib.pyplot as plt
%matplotlib
import numpy as np
from PIL import Image
import mrcfile
from numpy.fft import fft2, ifft2, fftfreq, fftshift

# Let's examine the mean diffraction space image, by taking the average over all probe positions:
diffraction_pattern_mean = np.mean(dataset.data, axis=(0,1))
py4DSTEM.visualize.show(diffraction_pattern_mean)

# Estimate the radius of the BF disk, and the center coordinates
probe_semiangle, qx0, qy0 = py4DSTEM.process.calibration.get_probe_size(diffraction_pattern_mean)

# plot the mean diffraction pattern, with the estimated probe radius overlaid as a circle
py4DSTEM.visualize.show_circles(diffraction_pattern_mean, (qx0, qy0), probe_semiangle)

# Print the estimate probe radius
print('Estimated probe radius =', '%.2f' % probe_semiangle, 'pixels')

# Next, create a bright field (BF) virtual detector using the the center beam position, and expanding the radius slightly (+ 2 px).
expand_BF = 15.0
image_BF = py4DSTEM.process.virtualimage.get_virtualimage(dataset, ((qx0, qy0), probe_semiangle + expand_BF))

# Show the BF image, next to the virtual detector we have used
py4DSTEM.visualize.show_circles(diffraction_pattern_mean, (qx0, qy0), probe_semiangle + expand_BF, figsize=(3,3))
py4DSTEM.visualize.show(image_BF,
                        min=7e7,
                        max=1e8)
plt.figure(), plt.imshow(image_BF, vmin=7.2e7, vmax=8.5e7)

# Generate the circular mask, using the same coordinates determined above.
mask = py4DSTEM.process.utils.make_circular_mask(shape = (dataset.Q_Nx,dataset.Q_Ny),
                                               qxy0 = (qx0, qy0),
                                               radius = probe_semiangle + expand_BF)
    
# Plot the mask
py4DSTEM.visualize.show(mask, figsize=(4,4))

# Calculate the center of mass for all probe positions
CoMx, CoMy = py4DSTEM.process.dpc.get_CoM_images(dataset, mask=mask)

# Plot the 2 CoM outputs, the x and y directions
py4DSTEM.visualize.show_image_grid(
    lambda i:[CoMx, CoMy][i],
    H=1,
    W=2,
    cmap='RdBu')

# Plot the gradient magnitude, from 0 to 2 standard deviations
py4DSTEM.visualize.show(np.sqrt(CoMx**2 + CoMy**2),
                       cmap='inferno',
                       clipvals='std',
                       min=0.0,
                       max=2.0)

# Use one of the py4DSTEM algorithms to check for:
# 1 - rotation offset between real and diffraction space coordinates
# 2 - a flip / transpose of the two spaces
theta, flip =  py4DSTEM.process.dpc.get_rotation_and_flip_maxcontrast(CoMx, CoMy, 360)

# Solve for minimum rotation from (-pi,pi) radian range
theta = np.mod(theta + np.pi, 2*np.pi) - np.pi
theta_deg = theta*180/np.pi
# Print the results
print('Image flip detected =', flip);
print('Best fit rotation = ', '%.4f' % theta_deg, 'degrees')


""""Reconstruct the phase contrast signal"""
# Input parameters
regLowPass = 0.0
regHighPass = 0.0
# Reconstruct the phase
phase, error = py4DSTEM.process.dpc.get_phase_from_CoM(
    CoMx, 
    CoMy, 
    theta=theta, 
    flip=flip,
    regLowPass=regLowPass, 
    regHighPass=regHighPass, 
    paddingfactor=2, # This parameter handles the boundary conditions via padding the reconstruction space
    stepsize=1, # Step size for each iteration
    n_iter=8)  # Number of iterations

# Plot the output phase image
py4DSTEM.visualize.show(phase, cmap='inferno', clipvals='manual',min=-200,max=230)
# Use Fourier interpolation to upsample the reconstructed phase image.
# phase_upsample = py4DSTEM.process.utils.fourier_resample(phase, output_size=(400, 500))
phase_upsample = py4DSTEM.process.utils.fourier_resample(phase, scale=8)

# plot the upsampled image
py4DSTEM.visualize.show(phase_upsample,
                       cmap='inferno',
                       clipvals='manual',
                       min=-200,
                       max=230)
