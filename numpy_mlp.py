import numpy as np
import random
from utilities import preprocess_dataset_for


class MLP ():
    #input layer is not counted in n_layers
    #output layer is
    def __init__(self, input_dim=2304, n_layers=5, hidden_dim=32, n_classes=7):
        rng = np.random.default_rng(seed=42) 
        
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.layers = []
        for i in range (n_layers):
            in_dim = hidden_dim
            out_dim = hidden_dim
            if i == 0:
                in_dim = input_dim
            elif (i+1) == n_layers:
                out_dim = n_classes
            std = 0.2
            if in_dim == out_dim:
                std = np.sqrt(2/in_dim)
            current_layer = rng.normal(
                #mean
                loc=0.0,      
                #standard deviation
                scale=std,        
                size=(in_dim, out_dim)
            ).astype(np.float32)
            self.layers.append (current_layer)

    def ReLU (self, x):
        x = np.asarray(x)
        return np.maximum(0, x)
    """
    def _get_normalized_logits_with_softmax_denom(self, logits):
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        softmax_denominator = np.sum(exp_logits)
        return exp_logits / softmax_denominator, softmax_denominator
    """
    def _get_normalized_logits_with_softmax_denom(self, logits):
        softmax_denominator = np.sum(np.exp(logits))
        return np.exp(logits) / softmax_denominator, softmax_denominator

    def forward(self, inputs, target_value=None, requires_grad=False):
        hidden_layer_activations = []
        #layer_activations.append(inputs.copy())
     
        
        for i in range(self.n_layers):
            #ReLU activations in all the layers but the last
            if i == 0:
                hidden_layer_activations.append(self.ReLU(inputs @ self.layers[i]))
            elif (i+1) < self.n_layers:
                hidden_layer_activations.append(self.ReLU(hidden_layer_activations[i-1] @ self.layers[i]))
            #no activation function applied so far
            #in the last layer
            #softmax will be applied to the output 
            #of the last layer later if necessary
            else:
                logits = hidden_layer_activations[i-1] @ self.layers[i]

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
            layer_gradients = [None] * self.n_layers

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
    print (f"Number of layers = {model.n_layers}")
    print (f"Dimensionality of hidden layers = {model.hidden_dim}")
    print ("_" * 50)
    train_loss_history = []
    val_loss_history = []

    for epoch_index in range(1, n_epochs + 1):

        print (f"Epoch {epoch_index}/{n_epochs}")
        print (f"current SGD learning rate = {lr}")
        total_train_loss = 0

        #shuffle training set in a reproducible manner
        random.seed(42 + epoch_index)
        random.shuffle(training_set) 

        for x,y in training_set:

            #get the CEL gradient from the forward pass directly
            loss, layer_grads = model.forward(x, y, requires_grad=True)
         
            #do SGD step
            for i in range(model.n_layers):
                #grad_norm = np.linalg.norm(layer_grads[i], ord=2)
                #if grad_norm > 1:
                #    layer_grads[i] /= grad_norm
                model.layers[i] -= lr*layer_grads[i]

            total_train_loss += loss
          
        #decrease lr each epoch
        lr *= sgd_lr_multiplier
        
        #compute, print and save avg loss per epoch
        avg_train_loss = total_train_loss / len(training_set)
        print (f"average train loss = {avg_train_loss:.5f}")
        train_loss_history.append(avg_train_loss)
 
        avg_val_loss = evaluate_model_on (model, validation_set)
        print (f"average val loss = {avg_val_loss:.5f}")
        val_loss_history.append(avg_val_loss)
        print ("_" * 50)

    return model, train_loss_history, val_loss_history


train_set, val_set, test_set = preprocess_dataset_for("mlp")

#numpy printing instruction for decimal notation
np.set_printoptions(
    precision   = 5,       
    floatmode   = 'fixed',  
    suppress    = True     
)


mlp = MLP()

SGD_LEARNING_RATE = 2e-3
LEARNING_RATE_MULTIPLIER_PER_EPOCH = 0.99
N_EPOCHS = 10
mlp, train_loss_history_SGD, val_loss_history_SGD = train_model_with_SGD (mlp,
                                            list(train_set),
                                            list(val_set),
                                            SGD_LEARNING_RATE,
                                            N_EPOCHS,
                                            LEARNING_RATE_MULTIPLIER_PER_EPOCH
                                            )
avg_test_loss = evaluate_model_on(mlp, list(test_set))
print (f"TEST LOSS = {avg_test_loss:.5f}")
print ("_" * 50)