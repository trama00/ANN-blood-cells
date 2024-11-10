from sklearn.model_selection import train_test_split
from tensorflow import keras as tfk
import numpy as np

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
    
def split_and_print_distribution(images, labels, val_size=0.2, test_size=0.2, seed=42):
    # Split data into train+val and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    # Split train+val into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=seed, stratify=y_train_val
    )

    # Print set sizes
    print("Data Set Sizes:")
    print(f"{'-'*20}")
    print(f"Train:      {X_train.shape[0]:>6}")
    print(f"Validation: {X_val.shape[0]:>6}")
    print(f"Test:       {X_test.shape[0]:>6}\n")

    # Print class distributions for each set
    print_class_distribution(y_train, "Train")
    print_class_distribution(y_val, "Validation")
    print_class_distribution(y_test, "Test")

    return X_train, X_val, X_test, y_train, y_val, y_test