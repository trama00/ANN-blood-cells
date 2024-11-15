import numpy as np
import tensorflow as tf
from multiprocessing import Pool
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from tensorflow.keras.applications import (
    Xception, VGG16, VGG19, ResNet50, ResNet50V2, ResNet101, ResNet101V2,
    ResNet152, ResNet152V2, InceptionV3, InceptionResNetV2, MobileNet,
    MobileNetV2, DenseNet121, DenseNet169, DenseNet201, NASNetMobile,
    NASNetLarge, EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,
    EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7,
    EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2, EfficientNetV2B3,
    EfficientNetV2S, EfficientNetV2M, EfficientNetV2L, ConvNeXtTiny,
    ConvNeXtSmall, ConvNeXtBase, ConvNeXtLarge, ConvNeXtXLarge
)



# Function to check if an image matches the conditions
def check_image(i, img, shrek, troll, tol):
    """Check if an image matches the shrek or troll images within a tolerance."""
    if (np.allclose(img, shrek, atol=tol) or np.allclose(img, troll, atol=tol)):
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



def get_base_model(model_name, input_shape):
    models = {
        'Xception': Xception, 'VGG16': VGG16, 'VGG19': VGG19, 'ResNet50': ResNet50,
        'ResNet50V2': ResNet50V2, 'ResNet101': ResNet101, 'ResNet101V2': ResNet101V2,
        'ResNet152': ResNet152, 'ResNet152V2': ResNet152V2, 'InceptionV3': InceptionV3,
        'InceptionResNetV2': InceptionResNetV2, 'MobileNet': MobileNet, 'MobileNetV2': MobileNetV2,
        'DenseNet121': DenseNet121, 'DenseNet169': DenseNet169, 'DenseNet201': DenseNet201,
        'NASNetMobile': NASNetMobile, 'NASNetLarge': NASNetLarge, 'EfficientNetB0': EfficientNetB0,
        'EfficientNetB1': EfficientNetB1, 'EfficientNetB2': EfficientNetB2, 'EfficientNetB3': EfficientNetB3,
        'EfficientNetB4': EfficientNetB4, 'EfficientNetB5': EfficientNetB5, 'EfficientNetB6': EfficientNetB6,
        'EfficientNetB7': EfficientNetB7, 'EfficientNetV2B0': EfficientNetV2B0, 'EfficientNetV2B1': EfficientNetV2B1,
        'EfficientNetV2B2': EfficientNetV2B2, 'EfficientNetV2B3': EfficientNetV2B3, 'EfficientNetV2S': EfficientNetV2S,
        'EfficientNetV2M': EfficientNetV2M, 'EfficientNetV2L': EfficientNetV2L, 'ConvNeXtTiny': ConvNeXtTiny,
        'ConvNeXtSmall': ConvNeXtSmall, 'ConvNeXtBase': ConvNeXtBase, 'ConvNeXtLarge': ConvNeXtLarge,
        'ConvNeXtXLarge': ConvNeXtXLarge
    }
    
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' is not available. Choose from: {list(models.keys())}")
    
    base_model = models[model_name](include_top=False, input_shape=input_shape, weights='imagenet')
    base_model.trainable = True  # UNFREEZED the base model

    return base_model


def analyze_mixup_distribution(y_original, y_augmented):
    """
    Analyzes the distribution of pure and mixed classes in a dataset.
    
    Args:
        y_original (np.ndarray): Original one-hot encoded labels.
        y_augmented (np.ndarray): Augmented one-hot encoded labels (post-MixUp).
    """
    # Convert one-hot encoded labels to class indices
    classes_original = np.argmax(y_original, axis=1)
    classes_augmented = np.argmax(y_augmented, axis=1)

    # Determine the unique classes and their counts in original and augmented datasets
    unique_classes, original_counts = np.unique(classes_original, return_counts=True)
    unique_aug_classes, aug_counts = np.unique(classes_augmented, return_counts=True)
    
    # Find mixed samples (where probabilities are between 0 and 1 for any class)
    is_mixed = (y_augmented > 0) & (y_augmented < 1)
    mixed_indices = np.any(is_mixed, axis=1)
    pure_augmented_classes = classes_augmented[~mixed_indices]
    mixed_classes = classes_augmented[mixed_indices]

    # Distribution of pure classes in the augmented dataset
    _, pure_aug_counts = np.unique(pure_augmented_classes, return_counts=True)

    # Distribution of mixed classes
    mixed_class_counts = [np.sum(mixed_classes == i) for i in unique_classes]
    
    # Plot original, pure augmented, and mixed class distributions
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot for original class distribution
    axs[0].bar(unique_classes, original_counts, color='skyblue')
    axs[0].set_title("Original Class Distribution")
    axs[0].set_xlabel("Class")
    axs[0].set_ylabel("Count")

    # Plot for pure augmented class distribution
    axs[1].bar(unique_classes, pure_aug_counts, color='lightgreen')
    axs[1].set_title("Pure Augmented Class Distribution")
    axs[1].set_xlabel("Class")
    axs[1].set_ylabel("Count")

    # Plot for mixed class distribution
    axs[2].bar(unique_classes, mixed_class_counts, color='salmon')
    axs[2].set_title("Mixed Class Distribution")
    axs[2].set_xlabel("Class")
    axs[2].set_ylabel("Count")
    
    # Print distributions
    print("Original Class Distribution:", dict(zip(unique_classes, original_counts)))
    print("Pure Augmented Class Distribution:", dict(zip(unique_classes, pure_aug_counts)))
    print("Mixed Class Distribution:", dict(zip(unique_classes, mixed_class_counts)))
    
    plt.tight_layout()
    plt.show()
    
    plot_mixup_pair_distribution(y_augmented)
    
    
def plot_mixup_pair_distribution(y_augmented):
    """
    Plots a heatmap representing the frequency of MixUp pairs between different classes in one-hot encoded labels.

    Args:
        y_augmented: np.ndarray - Augmented labels with MixUp applied, in one-hot encoded format (shape: [num_samples, num_classes]).
    """
    num_classes = y_augmented.shape[1]
    pair_counts = np.zeros((num_classes, num_classes), dtype=int)

    # Iterate over each augmented label to identify mixed pairs
    for label in y_augmented:
        # Identify indices with non-zero values (these represent the classes involved in MixUp)
        mixed_indices = np.where(label > 0)[0]
        
        # Only consider pairs where two distinct classes are involved
        if len(mixed_indices) == 2:
            i, j = mixed_indices
            pair_counts[i, j] += 1
            pair_counts[j, i] += 1  # Mirror for symmetric pairs

    # Plot the heatmap of MixUp pairs
    plt.figure(figsize=(10, 8))
    sns.heatmap(pair_counts, annot=True, fmt="d", cmap="coolwarm", square=True, 
                xticklabels=[f"Class {i}" for i in range(num_classes)],
                yticklabels=[f"Class {i}" for i in range(num_classes)])
    plt.title("MixUp Pair Distribution Heatmap")
    plt.xlabel("Class Index")
    plt.ylabel("Class Index")
    plt.show()



def print_sample(images, labels, random=True):
    unique_labels = np.unique(labels)
    print("Unique labels in the dataset:", unique_labels)
    for label in unique_labels:
        count = np.sum(labels == label)
        print(f"Class {label}: {count} images")

    # Plot one random image per class
    num_classes = len(unique_labels)
    num_cols = 4  # You can adjust this for layout
    num_rows = (num_classes + num_cols - 1) // num_cols

    plt.figure(figsize=(15, 4 * num_rows))
    for i, label in enumerate(unique_labels):
        # Get all indices for the current label
        indices = np.where(labels == label)[0]
        
        # Select a random index for the label
        idx = np.random.choice(indices) if random else indices[0]
        
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(np.clip(images[idx] / 255.0, 0, 1))
        plt.title(f"Class {labels[idx]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    
def compute_class_weights(labels):
    """
    Compute class weights, excluding labels created by MixUp.
    Labels are one-hot encoded. Only pure labels (with a class probability of 1) are considered.
    """
    # Convert labels to numpy array if not already
    labels = np.array(labels)

    # Identify "pure" labels (those with exactly one class probability of 1)
    pure_labels_mask = np.max(labels, axis=1) == 1
    pure_labels = np.argmax(labels[pure_labels_mask], axis=1)  # Convert to class indices

    # Count total number of pure labels
    num_samples = len(pure_labels)

    # Get unique classes and their counts
    unique_classes, class_counts = np.unique(pure_labels, return_counts=True)

    # Print class counts
    for cls, count in zip(unique_classes, class_counts):
        print(f"Class {cls}: {count} pure samples")

    # Total number of classes
    num_classes = len(unique_classes)

    # Calculate class weights
    class_weight = {
        cls: num_samples / (num_classes * count)
        for cls, count in zip(unique_classes, class_counts)
    }

    print(f"Class weights: {class_weight}")

    return class_weight


def test_on_random_groups(model, test_images, test_labels, group_size, num_groups, num_classes=8):
    """
    Test the model on random groups of images, calculate metrics and average them.

    Parameters:
    - model: The trained keras model.
    - test_images: The test set images.
    - test_labels: The test set labels.
    - group_size: The number of images in each group.
    - num_groups: The number of random groups to select.
    - num_classes: Number of classes in the classification task (default is 8).
    
    Returns:
    - Average accuracy, precision, recall, F1, and ROC-AUC scores across the groups.
    """
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1_scores = []
    roc_auc_scores = []
    
    for _ in range(num_groups):
        # Select random indices for this group
        indices = random.sample(range(test_images.shape[0]), group_size)
        X_group = test_images[indices]
        y_group = test_labels[indices]

        # Predict class probabilities and get predicted classes
        test_predictions = model.predict(X_group, verbose=0)
        test_predictions_classes = np.argmax(test_predictions, axis=-1)

        # Extract ground truth classes
        test_gt = np.argmax(y_group, axis=-1)

        # Calculate metrics for this group
        test_accuracy = accuracy_score(test_gt, test_predictions_classes)
        test_precision = precision_score(test_gt, test_predictions_classes, average='weighted')
        test_recall = recall_score(test_gt, test_predictions_classes, average='weighted')
        test_f1 = f1_score(test_gt, test_predictions_classes, average='weighted')

        # Append metrics to lists
        all_accuracies.append(test_accuracy)
        all_precisions.append(test_precision)
        all_recalls.append(test_recall)
        all_f1_scores.append(test_f1)

        # Compute the ROC-AUC for each class
        y_group_binarized = label_binarize(test_gt, classes=range(num_classes))
        group_roc_auc = []
        for i in range(num_classes):
            try:
                roc_auc = roc_auc_score(y_group_binarized[:, i], test_predictions[:, i])
                group_roc_auc.append(roc_auc)
            except ValueError:
                group_roc_auc.append(np.nan)  # If there are not enough samples for the class
        roc_auc_scores.append(np.nanmean(group_roc_auc))  # Take mean of ROC-AUC scores for the group

    # Calculate the average of all metrics
    avg_accuracy = np.mean(all_accuracies)
    avg_precision = np.mean(all_precisions)
    avg_recall = np.mean(all_recalls)
    avg_f1 = np.mean(all_f1_scores)
    avg_roc_auc = np.mean(roc_auc_scores)

    print(f'Average Accuracy: {round(avg_accuracy, 4)}')
    print(f'Average Precision: {round(avg_precision, 4)}')
    print(f'Average Recall: {round(avg_recall, 4)}')
    print(f'Average F1 Score: {round(avg_f1, 4)}')
    print(f'Average ROC-AUC Score: {round(avg_roc_auc, 4)}')

    return avg_accuracy, avg_precision, avg_recall, avg_f1, avg_roc_auc