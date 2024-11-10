import numpy as np
from multiprocessing import Pool
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.utils as tfku

# Function to check if an image matches the conditions
def check_image(i, img, shrek, troll, tol):
    """Check if an image matches the shrek or troll images within a tolerance."""
    if (np.allclose(img, shrek, atol=tol) or np.allclose(img, troll, atol=tol)) and i != 11959 and i != 13559:
        return i
    return None

# Function to parallelize the process of finding the indices to remove
def parallel_index_removal(data, shrek, troll, tol=0.0001, num_workers=4):
    """Parallelize the removal of images that match 'shrek' or 'troll'."""
    with Pool(processes=num_workers) as pool:
        results = pool.starmap(check_image, [(i, img, shrek, troll, tol) for i, img in enumerate(data)])

    # Filter out None values (no matches) and return the indices to remove
    index_to_remove = [index for index in results if index is not None]
    
    return index_to_remove

# Main Function (example of usage)
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

def balance_classes(images, labels, target_class_size=2000, augmentation=None):
    """
    Balances the dataset by augmenting underrepresented classes with transformations.

    Parameters:
    - images: Input images (numpy array of shape (num_images, height, width, channels))
    - labels: Input labels (numpy array of shape (num_images,))
    - target_class_size: Target number of images for each class after balancing
    - augmentation: Keras ImageDataGenerator for augmentation. If None, defaults to standard augmentation.

    Returns:
    - Tuple of (balanced_images, balanced_labels) as numpy arrays
    """
    target_class_size = int(target_class_size)  # Ensure integer value
    
    # Default augmentation if none is provided
    if augmentation is None:
        augmentation = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.1,
            brightness_range=[0.8, 1.2],
            fill_mode='reflect'
        )

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
            
            generator = augmentation.flow(
                class_images,
                batch_size=min(len(class_images), samples_needed),
                shuffle=False
            )
            
            # Generate samples until we reach desired count
            generated_samples = 0
            while generated_samples < samples_needed:
                batch = next(generator)
                samples_to_add = min(len(batch), samples_needed - generated_samples)
                augmented_images_list.append(batch[:samples_to_add])
                augmented_labels_list.append(np.full(samples_to_add, cls))
                generated_samples += samples_to_add

    # Convert lists to numpy arrays if there are augmented samples
    if augmented_images_list:
        augmented_images = np.concatenate(augmented_images_list, axis=0)
        augmented_labels = np.concatenate(augmented_labels_list, axis=0)
        
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