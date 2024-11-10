import numpy as np
import tensorflow as tf
from multiprocessing import Pool

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


def remove_background(image, threshold=0.5):
    """
    Removes the background by masking out lighter pixels.
    Assumes that the subject (darker) is below the threshold and the background (lighter) is above.

    Parameters:
    - image: Input image tensor.
    - threshold: Intensity threshold to separate dark (subject) and light (background) regions.

    Returns:
    - The image with the background removed.
    """
    mask = tf.where(image < threshold, 1.0, 0.0)  # Create a mask where darker areas are 1 (subject)
    return image * mask  # Apply the mask to keep only the subject


# Define the check_and_rescale function
@tf.function
def check_and_rescale(image):
    # Check if the image is within the [0, 255] range
    if tf.reduce_max(image) > 1.0:  # If the max value is greater than 1, it hasn't been rescaled
        image = image / 255.0  # Rescale to [0, 1]
    return image