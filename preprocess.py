import numpy as np
from multiprocessing import Pool
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    - target_class_size: Target number of images for the underrepresented classes
    - augmentation: Keras ImageDataGenerator for augmentation. If None, defaults to standard augmentation.

    Returns:
    - Balanced images and labels
    """
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
    class_counts = {cls: len(np.where(labels == cls)[0]) for cls in np.unique(labels)}

    # Find underrepresented classes
    underrepresented_classes = [cls for cls, count in class_counts.items() if count < target_class_size]
    
    augmented_images = []
    augmented_labels = []

    # Apply augmentation only to the underrepresented classes
    for cls in underrepresented_classes:
        class_indices = np.where(labels == cls)[0]
        class_images = images[class_indices]

        # Create ImageDataGenerator for the current class (using the passed augmentation or default)
        generator = augmentation

        # Fit the generator on the images
        augmented_gen = generator.flow(class_images, batch_size=len(class_images), shuffle=False)

        # Generate augmented images
        augmented_class_images = next(augmented_gen)

        # Add augmented images and labels
        augmented_images.extend(augmented_class_images)
        augmented_labels.extend([cls] * len(augmented_class_images))

    # Convert to numpy arrays
    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)
    
    # Append augmented images and labels back to the original dataset
    images_balanced = np.concatenate((images, augmented_images), axis=0)
    labels_balanced = np.concatenate((labels, augmented_labels), axis=0)

    return images_balanced, labels_balanced