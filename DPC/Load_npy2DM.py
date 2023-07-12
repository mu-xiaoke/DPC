import DigitalMicrograph as DM

import numpy as np

imgArray = np.load(r"D:\3D NAND\3D NAND new\2023-5-10 Spectra 300 200kV\Empad -2023-5-10\GL7 SZ11 CA2,5mrad_50um 543pm 0,6pA CL2.3m 40us 150e_per_step\iDPC.npy")
img = DM.CreateImage(imgArray)
img.SetName('4D-iDPC')
img.ShowImage() 
