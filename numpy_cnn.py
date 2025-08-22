from utilities import preprocess_dataset_for
import numpy as np
import math
from numpy.lib.stride_tricks import sliding_window_view
import random
import matplotlib.pyplot as plt
import time
import copy
import datetime

def log_message(message):
    with open(filename, "a") as f:
        f.write(message + "\n")

def maxpool2d_grayscale(image, pool_size=2, stride=2):
    # If image has a single channel dimension, remove it
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    
    H, W = image.shape
  
    patches = sliding_window_view(image, (pool_size, pool_size))
    patches = patches[::stride, ::stride, :, :]
    pooled = patches.max(axis=(-2, -1))
    return pooled



class CNN ():
    #def __init__(self, in_dim: int = 48, out_dim: int=7, kernel_size: int = 5, n_channels: int = 2, n_conv_layers: int = 5):
    def __init__(self, in_dim: int = 24, out_dim: int=7, kernel_size: int = 3, n_channels: int = 1, n_chan_mult: int = 1.1, n_conv_layers: int = 3):
        
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
            n_channels = math.ceil(n_chan_mult*n_channels)
            #n_channels = math.trunc(1.1*n_channels)
            #if n_channels < 256:
            #    n_channels *= 2

        
        

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

        return cc2d
    
    def _transposed_cross_correlation_2D_of_with (self, error_signal, network_layer):

     
        _, _, kH, kW = network_layer.shape

        pad_h, pad_w = kH - 1, kW - 1
        # pad spatial axes only
        error_signal_padded = np.pad(error_signal, ((0,0), (pad_h,pad_h), (pad_w,pad_w)), mode='constant')

        
        picture_patches = sliding_window_view(error_signal_padded, (kH, kW), axis = (1,2))

        tcc2d = np.einsum("ohwkl, oikl->ihw", picture_patches, network_layer)


        return tcc2d
   
    def _ReLU (self, x):
        return np.maximum(x, 0, out=x)
    """
    def _get_normalized_logits_with_softmax_denom(self, logits):
        softmax_denominator = np.sum(np.exp(logits))
        return np.exp(logits) / softmax_denominator, softmax_denominator
    """
    def _get_normalized_logits_with_softmax_denom(self, logits):
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        softmax_denominator = np.sum(exp_logits)
        return exp_logits / softmax_denominator, softmax_denominator
   
    def forward (self, inputs, target_value=None, requires_grad = False):
        if len(inputs.shape) == 2:
            inputs = np.expand_dims(inputs, axis=0)
  
        layer_activations = [inputs]
        for i in range(self.n_conv):
            cross_correlation = self._cross_correlation_2D_of_with(self.layers[i], layer_activations[i])
            layer_activations.append(self._ReLU(cross_correlation))
         

        #push output into the fully conntected layer
        #to get as many logits as there are
        #target classes
        #print ("fc shape ", self.fc.shape)
        #print ("last conv flattened shape ", layer_activations[-1].flatten().shape)
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
            #CEL_value = -logits[target_value] + np.log(softmax_denom)
            # normalized logits for stable computation
            CEL_value = -np.log(normalized_logits[target_value])
            
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
       

            
            for i in range(self.n_conv-1, -1, -1):
                
                error_signal *= (layer_activations[i+1] > 0)

                current_conv_layer_grad = self._cross_correlation_2D_of_with(
                    error_signal, layer_activations[i], forward_pass=False
                )

         
                error_signal = self._transposed_cross_correlation_2D_of_with(
                    error_signal, self.layers[i]
                )

                conv_layer_gradients[i] = current_conv_layer_grad
                
            
            layer_gradients = conv_layer_gradients
            layer_gradients.append(fc_gradient)

            return CEL_value, layer_gradients
        


def evaluate_model_on(model, dataset):
    total_loss = 0
    for x, y in dataset:
        total_loss += model.forward(x, y, requires_grad=False)
    return total_loss/len(dataset)
     


def train_model_with_SGD (model, 
                         training_set,
                         validation_set,
                         lr: float, 
                         n_epochs: int, 
                         sgd_lr_multiplier: float = 0.95
                        ):
    

    print ("_" * 50)
    print (f"Initial LR = {lr}")
    print (f"LR multipliter per epoch = {sgd_lr_multiplier:.5f}")
    print (f"Number of conv layers = {model.n_conv}")
    print (f"Number of channels in conv layers = {model.n_channels[1:]}")
    print ("_" * 50)

    log_message ("_" * 50)
    log_message (f"Initial LR = {lr}")
    log_message (f"LR multipliter per epoch = {sgd_lr_multiplier:.5f}")
    log_message (f"Number of conv layers = {model.n_conv}")
    log_message (f"Number of channels in conv layers = {model.n_channels[1:]}")
    log_message ("_" * 50)

    train_loss_history = []
    val_loss_history = []

    best_avg_epoch_loss = 1000

    for epoch_index in range(1, n_epochs + 1):

        print (f"Epoch {epoch_index}/{n_epochs}")
        print (f"current SGD learning rate = {lr}")

        log_message (f"Epoch {epoch_index}/{n_epochs}")
        log_message (f"current SGD learning rate = {lr}")

        total_train_loss = 0

        #shuffle training set in a reproducible manner
        random.seed(42 + epoch_index)
        random.shuffle(training_set) 
        start1 = time.time()
        for x,y in training_set:
            #start2 = time.time()
            #get the CEL gradient from the forward pass directly
            loss, layer_grads = model.forward(x, y, requires_grad=True)
         
            #do SGD step
            model.fc -= lr*layer_grads[-1]
            
            model.layers = [layer - lr * grad for layer, grad in zip(model.layers, layer_grads[:-1])]
                 
            #for i in range(model.n_conv):
            #   model.layers[i] -= lr*layer_grads[i]
           

       
            #end = time.time()
            #rint(f"Elapsed time per SGD iter: {end - start2} seconds")
            total_train_loss += loss
        end = time.time()
        print(f"Elapsed time per epoch iter: {end - start1} seconds")
        
        #decrease lr each epoch
        lr *= sgd_lr_multiplier
        
        #compute, print and save avg loss per epoch
        avg_train_loss = total_train_loss / len(training_set)
        print (f"average train loss = {avg_train_loss:.5f}")
        log_message (f"average train loss = {avg_train_loss:.5f}")
        train_loss_history.append(avg_train_loss)
 
        avg_val_loss = evaluate_model_on (model, validation_set)

        if avg_val_loss < best_avg_epoch_loss:
            best_avg_epoch_loss = avg_val_loss
            best_model = copy.deepcopy(model)



        print (f"average val loss = {avg_val_loss:.5f}")
        log_message (f"average val loss = {avg_val_loss:.5f}")
        val_loss_history.append(avg_val_loss)
        print ("_" * 50)
        log_message ("_" * 50)




    if best_model is not None:
        return best_model, train_loss_history, val_loss_history
    else:
        return model, train_loss_history, val_loss_history
    #return model, train_loss_history, val_loss_history

            
       
            
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"cnn_grid_search_log_{timestamp}.txt"

with open(filename, "w") as f:
    f.write("=== New Log Start ===\n")
       

train_set, val_set, test_set = preprocess_dataset_for("cnn")

train_set = [(maxpool2d_grayscale(img), label) for img, label in train_set]
val_set   = [(maxpool2d_grayscale(img), label) for img, label in val_set]
test_set  = [(maxpool2d_grayscale(img), label) for img, label in test_set]




np.set_printoptions(threshold=np.inf)



#numpy printing instruction for decimal notation
np.set_printoptions(
    precision   = 5,       
    floatmode   = 'fixed',  
    suppress    = True     
)


image_height = train_set[0][0].shape[0]

cnn = CNN(in_dim = image_height)




SGD_LEARNING_RATE = 15e-3
LEARNING_RATE_MULTIPLIER_PER_EPOCH = 0.97
N_EPOCHS = 2
mlp, train_loss_history_SGD, val_loss_history_SGD = train_model_with_SGD (cnn,
                                            list(train_set),
                                            list(val_set),
                                            SGD_LEARNING_RATE,
                                            N_EPOCHS,
                                            LEARNING_RATE_MULTIPLIER_PER_EPOCH
                                            )
avg_test_loss = evaluate_model_on(mlp, list(test_set))
print (f"TEST LOSS = {avg_test_loss:.5f}")
log_message (f"TEST LOSS = {avg_test_loss:.5f}")
print ("_" * 50)
log_message ("_" * 50)

# Indices
indices1 = range(len(train_loss_history_SGD))  
indices2 = range(len(val_loss_history_SGD)) 

# Plot both
plt.plot(indices1, train_loss_history_SGD, marker='o', linestyle='-', label='train loss hist')
plt.plot(indices2, val_loss_history_SGD, marker='s', linestyle='--', label='val loss hist')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and val loss vs epochs')
plt.legend()
plt.grid(True)

plt.savefig(f"best_cnn_loss_history_{timestamp}.png", dpi=300, bbox_inches='tight')

plt.show()

print ("_" * 50)
