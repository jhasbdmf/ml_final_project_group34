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
        
        

        #sampling boundaries for uniform Xavier weight init for a fully connected layer
        #last convolutional layer has self.n_channels[-1] channels
        #with feature widheight and width = in_dim - n_conv * (kernel_size - 1)
        #next output layer has out dim parameters
        n_param_per_channel_last_conv_layer = (self.input_w - self.n_conv * (self.kernel_size - 1)) ** 2
        n_param_total_last_conv_layer = self.n_channels[-1] * n_param_per_channel_last_conv_layer
        uniform_boundary = np.sqrt(6/(n_param_total_last_conv_layer + self.output_dim))

        #add one fully connected layer on top of convolutional ones
        #to project output of successive convolutions onto
        #the vector space of dimensionality out_dim
        self.n_fc = 1
        self.fc = rng.uniform(-uniform_boundary, uniform_boundary, size=(n_param_total_last_conv_layer, self.output_dim))
        
    
    def cross_correlation_2D_of (self, input, kernels):
  
        #print (f"kernel shape = {kernels.shape}")
        picture_patches = sliding_window_view(input, kernels[0].shape[1:], axis = (1,2))
        cc2d = np.einsum("ihwkl, oikl->ohw", picture_patches, kernels)
        
        return cc2d
   
    def ReLU (self, x):
        return np.maximum(x, 0, out=x)
    
    def _get_normalized_logits_with_softmax_denom(self, logits):
        softmax_denominator = np.sum(np.exp(logits))
        return np.exp(logits) / softmax_denominator, softmax_denominator
       
    def forward (self, inputs, target_value=None, requires_grad = False):
        if len(inputs.shape) == 2:
            inputs = np.expand_dims(inputs, axis=0)

        layer_activations = [inputs]
        for i in range(self.n_conv):
            cross_correlation = self.cross_correlation_2D_of(layer_activations[i], self.layers[i])
            layer_activations.append(self.ReLU(cross_correlation))
         
            print (layer_activations[-1].shape)
            print ("_" * 50)
        
        #push output into the fully conntected layer
        #to get as many logits as there are
        #target classes
        print (self.fc.shape)
        print (layer_activations[-1].flatten().shape)
        logits =  layer_activations[-1].flatten() @ self.fc

        
        #if no target value is passed to the model.forward
        #then the model is in the inference mode
        #no logit/output normalization/softmax is necessary
        #in the inference mode
        #since one can base model prediction on the highest
        #unnormalized/unsoftmaxed logit
        if target_value is None:
            return logits
           
        else:
            #this is softmax
            #softmax denominator is returned by softmax 
            #on top of normalized logits to be
            #reused in computing CEL
            normalized_logits, softmax_denom = self._get_normalized_logits_with_softmax_denom(logits)

            
            #CEL is equal to -ln(exp(logits[target_value])/softmax_denom))
            #the formula below is algebraically equivalent to the one above
            CEL_value = -logits[target_value] + np.log(softmax_denom)
            
        #if no gradient is required,
        #then just return CEL
        if not requires_grad:
            return CEL_value
        

        #if a target value is passed to model.forward
        #and CEL is required
        #then the model is in the training mode
        #one needs to do softmax on the inputs
        #to pass softmaxed logits into 
        #CEL/cross-entropy loss
        else:

            #this one is the gradient of CEL
            #d_softmax = normalized_logits.copy()
            d_softmax = normalized_logits
            d_softmax[target_value] -= 1

            #initialize an list with as many empty
            #elements as there are hidden layers
            #i.e. param matrices
            layer_gradients = [None] * (self.n_conv + self.n_fc)
            """
            #this computes gradients of layer params
            for i in range(self.n_layers-1, -1, -1):
                #output softmax layer
                if i == (self.n_layers-1):
                    #this is the part of the gradient
                    #which is reused across layers
                    dynamic_gradient = d_softmax

                    #this one contrains gradient of i-th layer
                    layer_gradients[i] = np.outer(hidden_layer_activations[i-1], dynamic_gradient)
                else:
                    #this one multiplies dynamic gradient by the gradient of pre-activations
                    #gradient of pre-activations is equal to the corresponding weight matrix
                    #which us stored in self.layers[i+1]
                    dynamic_gradient = self.layers[i+1] @ dynamic_gradient
                    #gradient of hidden ReLU activation is equal to 1
                    #if that activation is equal to 1 and 0 otherwise 
                    relu_grad = (hidden_layer_activations[i]>0).astype(float)
                    #mulitply dynamic grad by activation gradient
                    dynamic_gradient *= relu_grad
                    #input layer
                    if i == 0:
                        layer_gradients[i] = np.outer(inputs, dynamic_gradient)
                    #neither output nor input layer
                    else:
                        layer_gradients[i] = np.outer(hidden_layer_activations[i-1], dynamic_gradient)
            return CEL_value, layer_gradients
            """
       
            
       

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
    print (cnn.forward(x, y))
    break

#print (cnn.n_conv)

#for i in cnn.layers:
#        print (i)
#        print (i.shape)
#        print ("_" * 50)

