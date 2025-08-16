import numpy as np

def z_normalize_images(images):
    mean = images.mean()
    std  = images.std()
    eps  = 1e-8
    return (images - mean) / (std + eps)

class MLP ():
    #input layer is not counted in n_layers
    #output layer is
    def __init__(self, input_dim=2304, n_layers=10, hidden_dim=64, n_classes=7):
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

            current_layer = rng.normal(
                #mean
                loc=0.0,      
                #standard deviation
                scale=0.2,        
                size=(in_dim, out_dim)
            ).astype(np.float32)
            self.layers.append (current_layer)

    def ReLU (self, x):
        x = np.asarray(x)
        return np.maximum(0, x)
 
    def _get_normalized_logits_with_softmax_denom(self, logits):
        softmax_denominator = np.sum(np.exp(logits))
        return np.exp(logits) / softmax_denominator, softmax_denominator

    def forward(self, inputs, target_value=None):
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
        if target_value == None:
            return logits
        #if a target value is passed to model.forward
        #then the model is in the training mode
        #one needs to do softmax on the inputs
        #to pass softmaxed logits into 
        #CEL/cross-entropy loss
        else:
            #this is softmax
            #softmax denominator is returned by softmax 
            #on top of normalized logits to be
            #reused in computing CEL
            normalized_logits, softmax_denom = self._get_normalized_logits_with_softmax_denom(logits)

            
            #CEL is equal to -ln(exp(logits[target_value])/softmax_denom))
            #the formula below is algebraically equivalent to the one above
            CEL_value = -logits[target_value] + np.log(softmax_denom)

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
                    dynamic_gradient = self.layers[i+1] @ dynamic_gradient
                    relu_grad = (hidden_layer_activations[i]>0).astype(float)
                    dynamic_gradient *= relu_grad
                    #input layer
                    if i == 0:
                        layer_gradients[i] = np.outer(inputs, dynamic_gradient)
                    #neither output nor input layer
                    else:
                        layer_gradients[i] = np.outer(hidden_layer_activations[i-1], dynamic_gradient)
            return CEL_value, layer_gradients

def train_model_with_SGD (model, 
                         training_set,
                         lr: float, 
                         n_epochs: int, 
                         sgd_lr_multiplier: float = 0.95
                        ):
    
    print (f"Initial LR = {lr}")
    print (f"LR multipliter per epoch = {sgd_lr_multiplier}")
    print (f"Number of layers = {model.n_layers}")
    print (f"Dimensionality of hidden layers = {model.hidden_dim}")
    loss_history = []

    for epoch_index in range(1, n_epochs + 1):

        print (f"Epoch {epoch_index}/{n_epochs}")
        print (f"current SGD learning rate = {lr}")
        total_loss = 0
       
        #random.shuffle(token_pairs) 
        for x,y in training_set:

            #get the CEL gradient from the forward pass directly
            loss, layer_grads = model.forward(x, y)
         
            #do SGD step
            for i in range(model.n_layers):
                model.layers[i] -= lr*layer_grads[i]

            total_loss += loss
          
        #decrease lr each epoch
        lr *= sgd_lr_multiplier
        
        #compute, print and save avg loss per epoch
        avg_loss = total_loss / len(training_set)
        print (f"average loss is {avg_loss}")
        loss_history.append(avg_loss)
        print ("_" * 50)

    return model, loss_history


train_x = np.load('train_mlp_x.npy') 
train_y = np.load('train_mlp_y.npy')
test_x  = np.load('test_mlp_x.npy')   
test_y  = np.load('test_mlp_y.npy')

np.random.seed(42)
perm = np.random.permutation(len(train_x))
train_x = train_x[perm]
train_y = train_y[perm]



# z_normalization of inputs
train_x  = z_normalize_images(train_x)  
test_x  = z_normalize_images(test_x)  

#create training and test set by zipping 
#corresponding inputs and targets
training_set = zip(train_x, train_y)
test_set = zip(test_x, test_y)

#numpy printing instruction for decimal notation
np.set_printoptions(
    precision   = 18,       
    floatmode   = 'fixed',  
    suppress    = True     
)


mlp = MLP()

SGD_LEARNING_RATE = 2e-3
LEARNING_RATE_MULTIPLIER_PER_EPOCH = 0.99
N_EPOCHS = 20
mlp, loss_history_SGD = train_model_with_SGD (mlp,
                                            list(training_set),
                                            SGD_LEARNING_RATE,
                                            N_EPOCHS,
                                            LEARNING_RATE_MULTIPLIER_PER_EPOCH
                                            )

