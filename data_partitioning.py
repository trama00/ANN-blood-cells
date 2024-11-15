import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
from functools import partial
from typing import Tuple, Union, Optional
import numpy.typing as npt


def split_and_balance_distribution(images, labels, val_size=0.16, test_size=0.2, seed=42, balance_sets="train", TEST=False):
    """
    Splits the dataset into train, validation, and test sets, balances specified sets, and ensures no data is lost.

    Parameters:
    - images: np.array, the image data.
    - labels: np.array, the labels for the image data.
    - val_size: float, proportion of the dataset to allocate to validation.
    - test_size: float, proportion of the dataset to allocate to testing.
    - seed: int, random seed for reproducibility.
    - balance_sets: str, specifies which sets to balance ("train", "val", "test", or combinations like "train and val").
    - TEST: bool, whether to create a test set or not.
    """
    total_images = len(images)
    print(f"Total dataset size: {total_images} images\n")
    
    # Split the dataset into train+val and test
    if TEST:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            images, labels, test_size=test_size, random_state=seed, stratify=labels
        )
    else:
        X_train_val, X_test, y_train_val, y_test = images, None, labels, None

    # Split train+val into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size / (1 - test_size if TEST else 1), random_state=seed, stratify=y_train_val
    )

    # Balance validation set (always balanced)
    X_val, y_val = balance_dataset(X_val, y_val, balance_percentage=1.0)

    # Parse the `balance_sets` argument
    balance_sets = {s.strip() for s in balance_sets.split(" and ")}

    # Balance training set and adjust test set if required
    if "train" in balance_sets:
        X_train_balanced, y_train_balanced, X_test_adjusted, y_test_adjusted = balance_dataset_full(X_train, y_train, X_test, y_test)
        X_train, y_train = X_train_balanced, y_train_balanced
        X_test, y_test = X_test_adjusted, y_test_adjusted

    if "test" in balance_sets and TEST and X_test is not None:
        X_test, y_test = balance_dataset(X_test, y_test, balance_percentage=1.0)

    # Print set sizes
    train_size = X_train.shape[0]
    val_size = X_val.shape[0]
    test_size = X_test.shape[0] if TEST else 0

    print(f"Images in dataset: {labels.shape[0]}\n")
    print("Dataset Split Sizes:")
    print(f"{'-'*30}")
    print(f"Train:      {train_size:>6} images ({train_size / total_images * 100:.2f}%)")
    print(f"Validation: {val_size:>6} images ({val_size / total_images * 100:.2f}%)")
    print(f"Test:       {test_size:>6} images ({test_size / total_images * 100:.2f}%)\n" if TEST else "Test:       Not created\n")

    # Print class distributions for each set
    print_class_distribution(y_train, "Train")
    print_class_distribution(y_val, "Validation")
    if TEST:
        print_class_distribution(y_test, "Test")

    return X_train, X_val, X_test, y_train, y_val, y_test



# Helper function to balance dataset by redistributing leftover data
def balance_dataset_full(X, y, X_test=None, y_test=None):
    """
    Balances the dataset and redistributes leftover samples to the test set.

    Parameters:
    - X, y: Training data and labels to be balanced.
    - X_test, y_test: Optional test data and labels to include leftovers.

    Returns:
    - Balanced X, y for training, and adjusted X_test, y_test.
    """
    class_counts = Counter(y)
    min_class_size = min(class_counts.values())

    X_balanced = []
    y_balanced = []
    leftovers_X = []
    leftovers_y = []

    for class_label in class_counts:
        # Get indices of the current class
        class_indices = np.where(y == class_label)[0]

        # Split into balanced and leftover samples
        balanced_indices = np.random.choice(class_indices, min_class_size, replace=False)
        leftover_indices = np.setdiff1d(class_indices, balanced_indices)

        X_balanced.append(X[balanced_indices])
        y_balanced.append(y[balanced_indices])
        leftovers_X.append(X[leftover_indices])
        leftovers_y.append(y[leftover_indices])

    # Concatenate balanced data
    X_balanced = np.concatenate(X_balanced)
    y_balanced = np.concatenate(y_balanced)
    
    # Shuffle the balanced data
    shuffle_indices = np.random.permutation(len(y_balanced))
    X_balanced = X_balanced[shuffle_indices]
    y_balanced = y_balanced[shuffle_indices]

    # Handle leftovers
    if X_test is not None and y_test is not None:
        X_test = np.concatenate([X_test, *leftovers_X])
        y_test = np.concatenate([y_test, *leftovers_y])
    else:
        X_test = np.concatenate(leftovers_X) if leftovers_X else None
        y_test = np.concatenate(leftovers_y) if leftovers_y else None

    return X_balanced, y_balanced, X_test, y_test


# Helper function to balance a single dataset
def balance_dataset(X, y, balance_percentage=1.0):
    class_counts = Counter(y)
    min_class_size = int(min(class_counts.values()) * balance_percentage)

    X_balanced = []
    y_balanced = []

    for class_label in class_counts:
        class_indices = np.where(y == class_label)[0]
        sampled_indices = np.random.choice(class_indices, min_class_size, replace=False)

        X_balanced.append(X[sampled_indices])
        y_balanced.append(y[sampled_indices])

    X_balanced = np.concatenate(X_balanced)
    y_balanced = np.concatenate(y_balanced)

    return X_balanced, y_balanced


# Helper function to print class distribution
def print_class_distribution(labels, set_name):
    class_counts = Counter(labels)
    print(f"{set_name} set class distribution:")
    for class_label, count in class_counts.items():
        print(f"  Class {class_label}: {count}")
    print()



def apply_mixup(X, y, alpha=0.2, factor=1.5, batch_size=128):
    """
    Apply Mixup augmentation to image dataset with parallel processing, ensuring balanced class pairs.
    """
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if len(X) != len(y):
        raise ValueError("X and y must have same length")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if factor <= 1:
        raise ValueError("factor must be greater than 1")
    
    # Number of additional samples needed
    target_size = int(len(X) * factor)
    additional_samples = target_size - len(X)
    
    # Ensure batch_size doesn't exceed dataset size
    batch_size = min(batch_size, len(X))

    # Split the dataset into classes
    class_indices = [np.where(np.argmax(y, axis=1) == i)[0] for i in range(8)]
    
    # Generate class pairs
    class_pairs = [(i, j) for i in range(8) for j in range(8) if i != j]
    print(f"Generating {additional_samples} additional samples using Mixup")
    
    # Calculate how many pairs of samples to generate for each class pair
    num_pairs = len(class_pairs)
    num_samples_per_pair = additional_samples // num_pairs
    print(f"Generating {num_samples_per_pair} samples per class pair")
    
    # Prepare for storing the mixed images and labels
    mixed_X_list = []
    mixed_y_list = []
    
    # Perform Mixup for each pair of classes
    for (class_1, class_2) in class_pairs:
        # Get indices for the class pair
        class_1_indices = class_indices[class_1]
        class_2_indices = class_indices[class_2]
        
        # Generate pairs of images and labels from the two classes
        class_1_images = X[class_1_indices]
        class_1_labels = y[class_1_indices]
        
        class_2_images = X[class_2_indices]
        class_2_labels = y[class_2_indices]
        
        # Generate the desired number of mixed samples for this class pair
        for _ in range(num_samples_per_pair):
            # Randomly select images from each class
            while True:
                idx_1 = np.random.choice(class_1_images.shape[0])
                idx_2 = np.random.choice(class_2_images.shape[0])
                if (idx_1 != idx_2):
                    break
            
            img_1 = class_1_images[idx_1]
            lab_1 = class_1_labels[idx_1]
            
            img_2 = class_2_images[idx_2]
            lab_2 = class_2_labels[idx_2]
            
            # Create a new lambda value for mixup
            while True:
                lambda_val = np.random.beta(alpha, alpha) * np.random.uniform(low=0.92, high=0.97)
                if (lambda_val != 0) and (lambda_val != 1) and (lambda_val > 0.18):
                    break
            
            # Apply MixUp
            mixed_X = lambda_val * img_1 + (1 - lambda_val) * img_2
            mixed_y = lambda_val * lab_1 + (1 - lambda_val) * lab_2
            
            mixed_X_list.append(mixed_X)
            mixed_y_list.append(mixed_y)
    
    # Convert lists to arrays
    mixed_X = np.array(mixed_X_list)
    mixed_y = np.array(mixed_y_list)
    
    # Combine original and new samples
    X_mixed = np.concatenate([X, mixed_X], axis=0)
    y_mixed = np.concatenate([y, mixed_y], axis=0)

    return X_mixed, y_mixed

# Helper function to print class distribution
def print_class_distribution(y, set_name):
    unique, counts = np.unique(y, return_counts=True)
    percentages = counts / counts.sum() * 100
    print(f"{set_name} Set Distribution:")
    print(f"{'Class':<10}{'Count':>10}{'Percentage':>15}")
    print(f"{'-'*35}")
    for cls, count, percentage in zip(unique, counts, percentages):
        print(f"{cls:<10}{count:>10}{percentage:>14.2f}%")
    print()
