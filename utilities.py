def z_normalize_images(images):
    mean = images.mean()
    std  = images.std()
    eps  = 1e-8
    return (images - mean) / (std + eps)