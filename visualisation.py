import numpy as np
import matplotlib.pyplot as plt


X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")


def display_images(X, y, n_images=5):
    indices = np.random.choice(len(X), n_images, replace=False)
    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(indices):
        plt.subplot(1, n_images, i + 1)
        plt.imshow(X[idx])
        plt.title(f"Label: {y[idx]}")
        plt.axis('off')
    plt.show()


display_images(X_train, y_train, n_images=5)
