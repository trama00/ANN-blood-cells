import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras_cv as kcv
from tensorflow.keras.applications.convnext import preprocess_input
import random
from tensorflow.keras.utils import set_random_seed


# ---------------------------------------------------------------------- #
# Custom Functions to apply Augmentations and plot augmented images  
# ---------------------------------------------------------------------- #

def plot_original_images(data, indices):
    """
    Plots images based on their indices in the dataset and displays their position and label name.
    The layout and spacing are identical to the first row of apply_single_augmentations.

    Parameters:
        data (dict): Dictionary containing 'images' and 'labels'.
        indices (list of int): Positions of the images in the dataset.
    """

    # Define label-to-name mapping inside the function
    label_names = {
        0: "basophil",       
        1: "eosinophil",     
        2: "erythroblast",   
        3: "ig",             
        4: "lymphocyte",    
        5: "monocyte",      
        6: "neutrophil",     
        7: "platelet"        
    }
    
    X = data['images']  # Extract images from the dataset dictionary
    y = data['labels']  # Extract labels from the dataset dictionary
    
    images_set = [X[i] for i in indices]  # Select images based on specified indices
    labels = [y[i] for i in indices]      # Select corresponding labels based on indices
    
    num_images = len(images_set)  
    
    # Adjust figure size and layout to match the reference function
    plt.figure(figsize=(1.5 * num_images, 2.2))  
    plt.suptitle("Original Images", fontsize=14)  
    
    # Loop through the images, dataset indices, and labels
    for idx, (original_image, dataset_idx, label) in enumerate(zip(images_set, indices, labels)):
        label_name = label_names.get(label, "Unknown")  
        
        # Plot the original image horizontally
        plt.subplot(1, num_images, idx + 1)  
        plt.imshow(original_image.astype("uint8"))  
        plt.axis('off')  
        plt.title(f"Image {dataset_idx}, {label_name}", fontsize=8)  
    
    # Match padding and spacing
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)  
    plt.subplots_adjust(top=0.75)  
    plt.show()  


def apply_single_augmentations(data, Augmentations, indices, single_augmentations):
    """
    Applies single augmentations to a set of images, plots the original images with each augmented
    version next to them, without displaying labels.

    The layout and spacing are designed to be consistent with the plot_original_images function.

    Parameters:
        data (dict): Dictionary containing 'images' and 'labels'.
        indices (list of int): Positions of the images in the dataset.
        single_augmentations (list of str): List of augmentation names to be applied.
    """

    X = data['images']  # Extract images from the dataset dictionary
    y = data['labels']  # Extract labels from the dataset dictionary

    images_set = [X[i] for i in indices]  
    labels = [y[i] for i in indices]      

    num_images = len(images_set)         
    num_augmentations = len(single_augmentations) 

    # Calculate figure size to maintain consistent image size
    plt.figure(figsize=(1.5 * (num_augmentations + 1), 2.2 * num_images))
    plt.suptitle("Single Augmented Images", fontsize=14)  

    # Loop through each image, its index, and its label (labels not used)
    for idx, (original_image, dataset_idx, label) in enumerate(zip(images_set, indices, labels)):

        ax_original = plt.subplot(num_images, num_augmentations + 1, idx * (num_augmentations + 1) + 1)
        ax_original.imshow(original_image.astype("uint8"))  # Display the original image, converting to uint8 if necessary
        ax_original.axis('off')  # Turn off axis ticks and labels for a cleaner look
        ax_original.set_title(f"Image {dataset_idx}", fontsize=8)  # Set the title with image index only

        # Apply and plot each augmentation
        for aug_idx, aug_name in enumerate(single_augmentations):
      
            subplot_position = idx * (num_augmentations + 1) + 2 + aug_idx 
            if aug_name not in Augmentations:
                print(f"Error: Augmentation '{aug_name}' not found. Skipping.")
                continue 

            augmentation_layer = Augmentations[aug_name]  # Retrieve the augmentation function or layer
            augmented_image = augmentation_layer(original_image[None, ...]).numpy()[0]

            # Create a subplot for the augmented image
            ax_augmented = plt.subplot(num_images, num_augmentations + 1, subplot_position)
            ax_augmented.imshow(augmented_image.astype("uint8"))  
            ax_augmented.axis('off') 
            ax_augmented.set_title(f"{aug_name}", fontsize=8)  

    # Adjust layout to ensure consistent spacing and prevent overlap
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)  
    plt.subplots_adjust(top=0.95) 
    plt.show()  
        

def apply_inter_augmentations(data, Augmentations, indices, inter_augmentations):
    """
    Applies inter-augmentations (e.g., cutmix, mixup) to selected images,
    plots the original images, their indices, one-hot encoded labels (fractions included),
    and the augmented versions with their updated labels.

    Parameters:
        data (dict): Dictionary containing 'images' and 'labels'.
        indices (list of int): Indices of the images to be used for inter-augmentations.
        inter_augmentations (list of str): List of inter-augmentation names to be applied.
    """

    num_classes = 8  

    # Retrieve images and labels based on indices
    images_set = [data['images'][i] for i in indices]
    labels_set = [data['labels'][i] for i in indices]

    # Ensure at least two images are provided
    if len(images_set) < 2:
        raise ValueError("Inter-augmentations require at least two images.")

    # Convert images and labels to TensorFlow tensors
    images_tensor = tf.convert_to_tensor(images_set, dtype=tf.float32)
    labels_tensor = tf.keras.utils.to_categorical(labels_set, num_classes=num_classes).astype('float32')
    labels_tensor = tf.convert_to_tensor(labels_tensor)

    # Create a batch of images and labels
    image_set = {"images": images_tensor, "labels": labels_tensor}

    plt.figure(figsize=(12, 6))  
    plt.suptitle("Inter-Images Augmentations", fontsize=16)

    for i, aug_name in enumerate(inter_augmentations):
        # Get the augmentation layer
        if aug_name not in Augmentations:
            print(f"Error: Augmentation '{aug_name}' not found. Skipping.")
            continue

        aug_layer = Augmentations[aug_name]

        # Apply the augmentation
        augmented = aug_layer({"images": image_set["images"], "labels": image_set["labels"]})
        augmented_images = augmented["images"].numpy().astype('uint8')
        augmented_labels = augmented["labels"].numpy()

        # Plot original images, indices, and labels
        for j, (original_image, one_hot_label, idx) in enumerate(zip(images_set, labels_tensor.numpy(), indices)):
            plt.subplot(3, len(inter_augmentations) + 1, j * (len(inter_augmentations) + 1) + 1)
            plt.imshow(original_image.astype("uint8"))
            plt.axis('off')
            # Removed "Label:" from the title
            plt.title(f"Image {idx}\n{np.round(one_hot_label, 2)}", fontsize=8)

        # Plot augmented images and their labels
        for j, (aug_image, aug_label) in enumerate(zip(augmented_images, augmented_labels)):
            plt.subplot(3, len(inter_augmentations) + 1, j * (len(inter_augmentations) + 1) + 2 + i)
            plt.imshow(aug_image)
            plt.axis('off')
            plt.title(f"{aug_name} {j+1}\n{np.round(aug_label, 2)}", fontsize=8)

    plt.tight_layout(pad=1.0, w_pad=0.05, h_pad=1.0)
    plt.subplots_adjust(top=0.85) 
    plt.show()


def apply_pipeline_augmentations(data, augmentations, indices, pipelines, augpi, rate, seed):
    """
    Applies multiple augmentation pipelines dynamically selected from predefined augmentations,
    plots the original images with their indices and augmented versions,
    and includes the name of the pipeline applied in the title of the augmented images.

    Parameters:
        data (dict): Dictionary containing 'images' and 'labels'.
        indices (list of int): Indices of the images to be used for augmentation.
        pipelines (list of list of str): List of augmentation layer names to include in each pipeline.
        augpi (int): Number of augmentation layers to apply in sequence within each pipeline.
        rate (float): Probability rate of applying each augmentation layer.
        seed (int): Seed value for randomization to ensure reproducibility.
    """
    # Set random seed for reproducibility
    set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    X = data['images']
    y = data['labels']

    images_set = [X[i] for i in indices]
    labels = [y[i] for i in indices]

    num_images = len(images_set)
    num_pipelines = len(pipelines)

    # Validate augmentation layers
    valid_pipelines = []
    for pipeline_idx, selected_layers in enumerate(pipelines, start=1):
        valid_layers = [augmentations[layer] for layer in selected_layers if layer in augmentations]
        if not valid_layers:
            print(f"Warning: No valid layers found for Pipeline {pipeline_idx}. Skipping this pipeline.")
            continue
        valid_pipelines.append((valid_layers, f"Pipeline {pipeline_idx}"))

    if not valid_pipelines:
        print("Error: No valid augmentation pipelines to apply.")
        return

    plt.figure(figsize=(1.5 * (len(valid_pipelines) + 1), 2.2 * num_images))
    plt.suptitle("Pipeline Augmentations", fontsize=14)

    for img_idx, (original_image, dataset_idx) in enumerate(zip(images_set, indices)):
        subplot_position = img_idx * (len(valid_pipelines) + 1) + 1
        ax_original = plt.subplot(num_images, len(valid_pipelines) + 1, subplot_position)
        ax_original.imshow(original_image.astype("uint8"))
        ax_original.axis('off')
        ax_original.set_title(f"Image {dataset_idx}", fontsize=8)

        for pipeline_idx, (layers, pipeline_name) in enumerate(valid_pipelines):
            augmented_image = original_image.copy()
            for layer in random.sample(layers, k=min(augpi, len(layers))):
                if random.random() < rate:
                    augmented_image = layer(augmented_image[None, ...]).numpy()[0]

            subplot_position = img_idx * (len(valid_pipelines) + 1) + 2 + pipeline_idx
            ax_augmented = plt.subplot(num_images, len(valid_pipelines) + 1, subplot_position)
            ax_augmented.imshow(augmented_image.astype("uint8"))
            ax_augmented.axis('off')
            ax_augmented.set_title(pipeline_name, fontsize=8)

    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plt.subplots_adjust(top=0.95)
    plt.show()

