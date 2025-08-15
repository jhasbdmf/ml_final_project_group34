import numpy as np

def z_normalize_images(images):
    mean = images.mean()
    std  = images.std()
    eps  = 1e-8
    return (images - mean) / (std + eps)

class MLP ():
    def __init__(self, input_dim=2304, hidden_dim=32, num_classes=7):
        rng = np.random.default_rng(seed=42) 
        self.layer1 = rng.normal(
            #mean
            loc=0.0,      
            #standard deviation
            scale=0.2,        
            size=(input_dim, hidden_dim)
        ).astype(np.float32)
        self.layer2 = rng.normal(
            loc=0.0,          
            scale=0.2,       
            size=(hidden_dim, num_classes)
        ).astype(np.float32)

    def ReLU (self, x):
        x = np.asarray(x)
        return np.maximum(0, x)
 
    def _get_normalized_logits_with_softmax_denom(self, logits):
        softmax_denominator = np.sum(np.exp(logits))
        return np.exp(logits) / softmax_denominator, softmax_denominator

    def forward(self, inputs, target_value=None):
        hidden_activations = self.ReLU(inputs @ self.layer1)
        logits = hidden_activations @ self.layer2

        if target_value == None:
            return logits
        else:
            normalized_logits, softmax_denom = self._get_normalized_logits_with_softmax_denom(logits)

            CEL_value = -logits[target_value] + np.log(softmax_denom)

            normalized_logits[target_value] -= 1

            dW2 = np.outer(hidden_activations, normalized_logits)
            relu_grad = (hidden_activations > 0).astype(float)
            dW1 = np.outer(inputs, (self.layer2 @ normalized_logits)*relu_grad)
            
            return CEL_value, dW2, dW1


def train_model_with_SGD (model, 
                         training_set,
                         lr: float, 
                         n_epochs: int, 
                         sgd_lr_multiplier: float = 0.95
                        ):
    
    loss_history = []

    for epoch_index in range(1, n_epochs + 1):

        print (f"Epoch {epoch_index}/{n_epochs}")
        print (f"current SGD learning rate = {lr}")
        total_loss = 0
       
        #random.shuffle(token_pairs) 
        for x,y in training_set:

            #get the CEL gradient from the forward pass directly
            loss, dW2, dW1 = model.forward(x, y)
         
            model.layer1 -= lr*dW1
            model.layer2 -= lr*dW2
            total_loss += loss
          
   
        lr *= sgd_lr_multiplier
        
        

        total_loss /= len(training_set)
        print (f"average loss is {total_loss}")
        loss_history.append(total_loss)
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
print (f"LR multipliter per epoch = {LEARNING_RATE_MULTIPLIER_PER_EPOCH}")
N_EPOCHS = 10
mlp, loss_history_SGD = train_model_with_SGD (mlp,
                                            list(training_set),
                                            SGD_LEARNING_RATE,
                                            N_EPOCHS,
                                            LEARNING_RATE_MULTIPLIER_PER_EPOCH
                                            )

