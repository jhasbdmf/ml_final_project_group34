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

        #no more conv layers are initialized 
        #than necessary so that the receptive field
        #of the last conv layer covers the entire input image
        self.n_conv = min(n_conv_layers, (in_dim // (kernel_size-1)) -1 )

        self.layers = []

        for i in range(self.n_conv):
            self.n_channels.append(n_channels)

            #compute standard deviation for Kaiming initialization of
            #weights of convolutional kernels
            #because ReLU activations are used
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
        
    
    #def cross_correlation_2D_of (self, input, kernels):
    def _cross_correlation_2D_of_with (self, kernels, input, forward_pass = True):
       
        picture_patches = sliding_window_view(input, kernels.shape[-2:], axis = (1,2))

        #this is cc2d for the backward pass pass
        if forward_pass:
            cc2d = np.einsum("ihwkl, oikl->ohw", picture_patches, kernels)
            
        #this is cc2d for the forward pass
        else:
            cc2d = np.einsum("ihwkl, okl->oihw", picture_patches, kernels)
        #if forward_pass:
        #    cc2d = np.einsum("ihwkl, oikl->ohw", picture_patches, kernels)
        #else:
     
        #    print (f"patches {picture_patches.shape}")
        #    print (f"current kernel {kernels.shape}")
        #    cc2d = np.einsum("ihwkl, okl->oihw", picture_patches, kernels)
        #    print (f"grad {cc2d.shape}")
        return cc2d
    
    def _transposed_cross_correlation_2D_of_with (self, error_signal, network_layer):
        #print (f"kernel shape = {kernels.shape}")
        #picture_patches = sliding_window_view(input, kernels[0].shape[-2:], axis = (1,2))
        #picture_patches = sliding_window_view(input, kernels[0].shape[-2:], axis = (2,3))

        error_signal_padded = np.pad(error_signal, ((0,0), (2,2), (2,2)), mode='constant')
        picture_patches = sliding_window_view(error_signal_padded, network_layer.shape[-2:], axis = (1,2))
        #print (f"pict patches shape = {picture_patches.shape}")
        #print (f"net layer shape = {network_layer.shape}")
        #print (f"error signal shape = {error_signal.shape}")
        tcc2d = np.einsum("ohwkl, oikl->ihw", picture_patches, network_layer)
        #print (f"tcc2d shape = {tcc2d.shape}")

        return tcc2d
   
    def _ReLU (self, x):
        return np.maximum(x, 0, out=x)
    
    def _get_normalized_logits_with_softmax_denom(self, logits):
        softmax_denominator = np.sum(np.exp(logits))
        return np.exp(logits) / softmax_denominator, softmax_denominator
       
    def forward (self, inputs, target_value=None, requires_grad = False):
        if len(inputs.shape) == 2:
            inputs = np.expand_dims(inputs, axis=0)

        layer_activations = [inputs]
        for i in range(self.n_conv):
            cross_correlation = self._cross_correlation_2D_of_with(self.layers[i], layer_activations[i])
            layer_activations.append(self._ReLU(cross_correlation))
         
            print (f"{i+1}th conv layer act shape = {layer_activations[-1].shape}")
            print (f"{i+1}th conv layer kernel shape = {self.layers[i].shape}")
       
            print ("_" * 50)
        
        #push output into the fully conntected layer
        #to get as many logits as there are
        #target classes
        print ("fc shape ", self.fc.shape)
        print ("last conv flattened shape ", layer_activations[-1].flatten().shape)
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

            #gradient of the fully connected layer
            fc_gradient = np.outer(layer_activations[-1].flatten(), d_softmax)

            #initialize an list with as many empty
            #elements as there are conv layers
            #i.e. cross-correlation param tensors
            conv_layer_gradients = [None] * (self.n_conv)
            #error_signal = self.fc @ d_softmax

            #!!!!!
            reshaped_fc_shape = list(layer_activations[-1].shape)
            reshaped_fc_shape.append(self.output_dim)
           
            error_signal = self.fc.reshape(reshaped_fc_shape) @ d_softmax
            #print ("error signal shape ", error_signal.shape)

            
            for i in range(self.n_conv-1, -1, -1):

                current_conv_layer_grad = self._cross_correlation_2D_of_with(error_signal, layer_activations[i], forward_pass=False)
                error_signal = self._transposed_cross_correlation_2D_of_with(error_signal, self.layers[i]) 
                conv_layer_gradients[i] = current_conv_layer_grad
                print ("current grad shape", current_conv_layer_grad.shape)
            
            layer_gradients = conv_layer_gradients
            layer_gradients.append(fc_gradient)

            return CEL_value, layer_gradients
            
       
            
       

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
    _, _ = cnn.forward(x, y, requires_grad=True)
    break

#print (cnn.n_conv)

#for i in cnn.layers:
#        print (i)
#        print (i.shape)
#        print ("_" * 50)

