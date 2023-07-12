# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 12:54:25 2023

@author: xiaoke.mu
"""
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
def CenterMap(data_4D, ref):
    ## ref = an array of the reference center
    Data_shape = data_4D.shape
    CenterX = np.zeros([Data_shape[0],Data_shape[1]])
    CenterY = np.zeros([Data_shape[0],Data_shape[1]])
    for i in range(Data_shape[0]):
        for j in range(Data_shape[1]):
            CenterX[i,j], CenterY[i,j] = center(data_4D[i,j])
    #Correct the center x and y using their average
    CenterX = CenterX-ref[0]
    CenterY = CenterY-ref[1]
    return [CenterX, CenterY]

min_int = 1
def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
def ewpc(diff_pattern):
    #return np.abs(fftshift(fft2(np.pad(diff_pattern, round(diff_pattern.shape[0]/2), pad_with))))
    #diff_pattern = np.pad(diff_pattern, round(diff_pattern.shape[0]/2), pad_with)
    #ceps_diff = abs(fftshift(fft2(np.log(abs((diff_pattern))+min_int))))
    ceps_diff = abs(fftshift(fft2(diff_pattern)))
    return ceps_diff
    
cp = ewpc(dataset.data)
mask_g1 = np.zeros((256,256),dtype=bool)
mask_g1[126:131,111:116]='true'
mask_g2 = np.zeros((256,256),dtype=bool)
mask_g2[134:134+5,126:126+5]='true'

g1_x, g1_y= CenterMap(cp*mask_g1,[128,128])
exx_global=(g1_x-np.average(g1_x))/np.average(g1_x)*100
plt.figure(),plt.imshow(exx_global)