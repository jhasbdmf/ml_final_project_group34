import os
import numpy as np
from PIL import Image

def build_and_save_datasets(split_dir,
                            prefix="train",
                            image_size=(48, 48),
                            seed=42):
    np.random.seed(seed)
    # Find class subfolders
    classes = sorted(
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    )
    # Gather image paths and labels
    paths, labels = [], []
    for class_idx, class_name in enumerate(classes):
        folder = os.path.join(split_dir, class_name)
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(folder, fname))
                labels.append(class_idx)
    N = len(paths)
    H, W = image_size
    cnn_x = np.empty((N, H, W), dtype=np.uint8)
    cnn_y = np.array(labels,     dtype=np.uint8)
    mlp_x = np.empty((N, H * W), dtype=np.uint8)
    mlp_y = cnn_y.copy()
    for i, path in enumerate(paths):
        img = Image.open(path)
        if img.mode != "L":
            img = img.convert("L")
        if img.size != (W, H):
            img = img.resize((W, H))
        arr = np.asarray(img, dtype=np.uint8)
        cnn_x[i] = arr
        mlp_x[i] = arr.ravel()
    # Shuffle in unison
    perm = np.random.permutation(N)
    cnn_x, cnn_y = cnn_x[perm], cnn_y[perm]
    mlp_x, mlp_y = mlp_x[perm], mlp_y[perm]
    # Save making the split explicit in the filename
    np.save(f"{prefix}_cnn_x.npy", cnn_x)
    np.save(f"{prefix}_cnn_y.npy", cnn_y)
    np.save(f"{prefix}_mlp_x.npy", mlp_x)
    np.save(f"{prefix}_mlp_y.npy", mlp_y)
    print(f"Saved {N} {prefix} samples to:")
    print(f"  • {prefix}_cnn_x.npy  →", cnn_x.shape)
    print(f"  • {prefix}_cnn_y.npy  →", cnn_y.shape)
    print(f"  • {prefix}_mlp_x.npy  →", mlp_x.shape)
    print(f"  • {prefix}_mlp_y.npy  →", mlp_y.shape)
    print(f"Classes: {classes}")

if __name__ == "__main__":
    build_and_save_datasets(
        split_dir="./dataset/train",
        prefix="train",
        image_size=(48, 48)
    )
    build_and_save_datasets(
        split_dir="./dataset/test",
        prefix="test",
        image_size=(48, 48)
    )