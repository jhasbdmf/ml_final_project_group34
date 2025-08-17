from utilities import preprocess_dataset_for
import numpy as np
import math
from numpy.lib.stride_tricks import sliding_window_view



class CNN ():
    def __init__(self, in_dim: int = 48, out_dim: int=7, kernel_size: int = 3, n_channels: int = 16, n_conv_layers: int = 25):
        
        rng = np.random.default_rng(42)

        self.input_h = in_dim
        self.input_w = in_dim
        self.output_dim = out_dim
        self.kernel_size = kernel_size
        #contains number of channels in i-th layer
        #where 0th layer is the input layer with
        #one channel since inputs are grayscale
        self.n_channels = [1]
        self.n_conv = min(n_conv_layers, (in_dim // (kernel_size-1)) -1 )
        self.layers = []

        for i in range(self.n_conv):
            self.n_channels.append(n_channels)

            #compute standard deviation for Kaiming initialization of
            #weights of convolutional kernels
            current_input_dim = n_channels * (kernel_size ** 2)
            std = np.sqrt(2/current_input_dim )

            #initialize as many kernels as there are channels
            #those n_channels kernales are layer parameters
            self.layers.append(rng.normal(loc=0.0, scale=std, size=(self.n_channels[i+1], self.n_channels[i], self.kernel_size, self.kernel_size)))

            #increase n_channels across layers gradually 
            #n_channels = math.ceil(1.1*n_channels)
            n_channels = math.trunc(1.1*n_channels)
    
    def cross_correlation_2D_of (self, input, kernels):
  
        #print (f"kernel shape = {kernels.shape}")
        picture_patches = sliding_window_view(input, kernels[0].shape[1:], axis = (1,2))
        cc2d = np.einsum("ihwkl, oikl->ohw", picture_patches, kernels)
        
        return cc2d
   
    def ReLU (self, x):
        return np.maximum(x, 0, out=x)
       
    def forward (self, inputs, targets=None):
        if len(inputs.shape) == 2:
            inputs = np.expand_dims(inputs, axis=0)

        layer_activations = [inputs]
        for i in range(self.n_conv):
            cross_correlation = self.cross_correlation_2D_of(layer_activations[i], self.layers[i])
            layer_activations.append(self.ReLU(cross_correlation))
         
            print (layer_activations[-1].shape)
            print ("_" * 50)
       

train_set, val_set, test_set = preprocess_dataset_for("cnn")



np.set_printoptions(threshold=np.inf)



#numpy printing instruction for decimal notation
np.set_printoptions(
    precision   = 5,       
    floatmode   = 'fixed',  
    suppress    = True     
)



cnn = CNN()

for x, y in train_set:
    cnn.forward(x)
    break

print (cnn.n_conv)

#for i in cnn.layers:
#        print (i)
#        print (i.shape)
#        print ("_" * 50)

