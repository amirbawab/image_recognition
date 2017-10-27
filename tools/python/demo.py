import numpy   as np 
import scipy.misc # to visualize only  
x = np.loadtxt("train_x.csv", delimiter=",") # load from text 
y = np.loadtxt("train_y.csv", delimiter=",") 
x = x.reshape(-1, 64, 64) # reshape 
y = y.reshape(-1, 1) 
scipy.misc.imshow(x[0]) # to visualize only 
