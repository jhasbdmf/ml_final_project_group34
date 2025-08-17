import numpy as np

def z_normalize_images(images):
    mean = images.mean()
    std  = images.std()
    eps  = 1e-8
    return (images - mean) / (std + eps)

def preprocess_dataset_for (ann_type: str):
    train_x = np.load(f'train_{ann_type}_x.npy') 
    train_y = np.load(f'train_{ann_type}_y.npy')
    val_and_test_x  = np.load(f'test_{ann_type}_x.npy')   
    val_and_test_y  = np.load(f'test_{ann_type}_y.npy')

    N = len(val_and_test_x)
    perm = np.random.RandomState(seed=42).permutation(N)
    split_at = N // 2

    val_idx = perm[:split_at]
    test_idx = perm[split_at:]

    val_x  = val_and_test_x[val_idx]
    val_y  = val_and_test_y[val_idx]
    test_x = val_and_test_x[test_idx]
    test_y = val_and_test_y[test_idx]



    # z_normalization of inputs
    train_x  = z_normalize_images(train_x)  
    val_x  = z_normalize_images(val_x)  
    test_x  = z_normalize_images(test_x)  

    #create training, val and test set by zipping 
    #corresponding inputs and targets
    training_set = zip(train_x, train_y)
    val_set = zip(val_x, val_y)
    test_set = zip(test_x, test_y)

    return training_set, val_set, test_set 