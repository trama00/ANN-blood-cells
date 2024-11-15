import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomBrightness
from tensorflow.keras import Sequential
import tensorflow.keras.utils as tfku
from utils import parallel_index_removal


def one_hot_encode_labels(y_train, y_val, y_test):
    """
    One-hot encodes the provided label arrays.

    Parameters:
    - y_train: Array of training labels.
    - y_val: Array of validation labels.
    - y_test: Array of test labels.

    Returns:
    - Tuple of one-hot encoded arrays for y_train, y_val, and y_test.
    """
    y_train_encoded = tfku.to_categorical(y_train)
    y_val_encoded = tfku.to_categorical(y_val)
    if y_test is not None:
        y_test_encoded = tfku.to_categorical(y_test)
        return y_train_encoded, y_val_encoded, y_test_encoded

    else:
        return y_train_encoded, y_val_encoded


import numpy as np

def clean_dataset(images, labels):
    # Define shrek and troll images
    shrek = images[11959]
    troll = images[13559]

    # Find indices of shrek and troll images to remove
    index_to_remove = parallel_index_removal(images, shrek, troll, tol=0.0001, num_workers=4)

    # Identify indices of non-unique images to remove
    _, unique_indices = np.unique(images, axis=0, return_index=True)
    all_indices = np.arange(len(images))
    duplicate_indices = np.setdiff1d(all_indices, unique_indices)
    
    # Combine duplicate indices with the shrek and troll indices
    index_to_remove = np.unique(np.concatenate((index_to_remove, duplicate_indices)))

    # Remove the images and labels at those indices
    images = np.delete(images, index_to_remove, axis=0)
    labels = np.delete(labels, index_to_remove)

    print(f"Removed {len(index_to_remove)} duplicate and unwanted images.")
    
    # Return the modified images and labels
    return images, labels
