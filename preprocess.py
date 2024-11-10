import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomBrightness
from tensorflow.keras import Sequential
import tensorflow.keras.utils as tfku
from utils import parallel_index_removal

def balance_classes(images, labels, target_class_size=2000, augmentation=None):
    """
    Balances the dataset by augmenting underrepresented classes with transformations.

    Parameters:
    - images: Input images (numpy array of shape (num_images, height, width, channels))
    - labels: Input labels (numpy array of shape (num_images,))
    - target_class_size: Target number of images for each class after balancing
    - augmentation: Optional keras Sequential model for augmentation. If None, defaults to standard augmentation.

    Returns:
    - Tuple of (balanced_images, balanced_labels) as numpy arrays
    """
    target_class_size = int(target_class_size)  # Ensure integer value

    # Default augmentation if none is provided
    if augmentation is None:
        augmentation = Sequential([
            RandomFlip("horizontal_and_vertical"),
            RandomRotation(0.2),
            RandomBrightness(0.2)
        ])

    # Count the occurrences of each class
    unique_classes = np.unique(labels)
    class_counts = {cls: np.sum(labels == cls) for cls in unique_classes}
    
    augmented_images_list = []
    augmented_labels_list = []

    # Apply augmentation to the underrepresented classes
    for cls in unique_classes:
        current_count = class_counts[cls]
        if current_count < target_class_size:
            # Calculate how many additional samples we need
            samples_needed = target_class_size - current_count
            
            # Get images for current class
            class_indices = np.where(labels == cls)[0]
            class_images = images[class_indices]
            
            generated_samples = 0
            while generated_samples < samples_needed:
                # Augment each image individually
                for img in class_images:
                    if generated_samples >= samples_needed:
                        break
                    augmented_img = augmentation(tf.expand_dims(img, axis=0))
                    augmented_images_list.append(augmented_img[0].numpy())
                    augmented_labels_list.append(cls)
                    generated_samples += 1

    # Convert lists to numpy arrays if there are augmented samples
    if augmented_images_list:
        augmented_images = np.array(augmented_images_list)
        augmented_labels = np.array(augmented_labels_list)
        
        # Combine original and augmented data
        images_balanced = np.concatenate((images, augmented_images), axis=0)
        labels_balanced = np.concatenate((labels, augmented_labels), axis=0)
    else:
        images_balanced = images
        labels_balanced = labels

    return images_balanced, labels_balanced


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
    y_test_encoded = tfku.to_categorical(y_test)
    
    return y_train_encoded, y_val_encoded, y_test_encoded


def clean_dataset(images, labels):
    # Define shrek and troll images
    shrek = images[11959]
    troll = images[13559]

    # Find indices to remove
    index_to_remove = parallel_index_removal(images, shrek, troll, tol=0.0001, num_workers=4)

    # Remove the images and labels at those indices
    images = np.delete(images, index_to_remove, axis=0)
    labels = np.delete(labels, index_to_remove)

    # Return the modified images and labels
    return images, labels