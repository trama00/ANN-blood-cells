from sklearn.model_selection import train_test_split
from tensorflow import keras as tfk
import numpy as np

import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

def split_and_print_distribution(images, labels, val_size=0.16, test_size=0.2, seed=42):
    # Split data into train+val and test sets (test_size will be ~20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    
    # Split train+val into train and validation sets (val_size will be ~16% of total data)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size / (1 - test_size), random_state=seed, stratify=y_train_val
    )

    # Balance the training and validation sets by undersampling the larger classes
    X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)
    X_val_balanced, y_val_balanced = balance_dataset(X_val, y_val)

    # Print set sizes
    print("Data Set Sizes:")
    print(f"{'-'*20}")
    print(f"Train:      {X_train_balanced.shape[0]:>6}")
    print(f"Validation: {X_val_balanced.shape[0]:>6}")
    print(f"Test:       {X_test.shape[0]:>6}\n")

    # Print class distributions for each set
    print_class_distribution(y_train_balanced, "Train")
    print_class_distribution(y_val_balanced, "Validation")
    print_class_distribution(y_test, "Test")

    return X_train_balanced, X_val_balanced, X_test, y_train_balanced, y_val_balanced, y_test


# Helper function to balance dataset by undersampling larger classes
def balance_dataset(X, y):
    class_counts = Counter(y)
    min_class_size = min(class_counts.values())  # The smallest class size
    
    X_balanced = []
    y_balanced = []
    
    for class_label in class_counts:
        # Get the indices of the current class
        class_indices = np.where(y == class_label)[0]
        
        # Randomly sample the smaller number of instances for this class
        sampled_indices = np.random.choice(class_indices, min_class_size, replace=False)
        
        X_balanced.append(X[sampled_indices])
        y_balanced.append(y[sampled_indices])
    
    # Concatenate to get the final balanced dataset
    X_balanced = np.concatenate(X_balanced)
    y_balanced = np.concatenate(y_balanced)
    
    return X_balanced, y_balanced


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
