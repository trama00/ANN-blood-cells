import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
from functools import partial

def split_and_print_distribution(images, labels, val_size=0.16, test_size=0.2, mixup_alpha=0.2, seed=42):
    # Split data into train+val and test sets (test_size will be ~20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    
    # Split train+val into train and validation sets (val_size will be ~16% of total data)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size / (1 - test_size), random_state=seed, stratify=y_train_val
    )
    
    # Balance the test and validation datasets
    X_test_balanced, y_test_balanced = balance_dataset(X_test, y_test, balance_percentage=1.0)
    X_val_balanced, y_val_balanced = balance_dataset(X_val, y_val, balance_percentage=1.0)
    
    # Print set sizes
    print("Data Set Sizes:")
    print(f"{'-'*20}")
    print(f"Train:      {X_train.shape[0]:>6}")
    print(f"Validation: {X_val_balanced.shape[0]:>6}")
    print(f"Test:       {X_test_balanced.shape[0]:>6}\n")

    # Print class distributions for each set
    print_class_distribution(y_train, "Train")
    print_class_distribution(y_val_balanced, "Validation")
    print_class_distribution(y_test_balanced, "Test")

    return X_train, X_val_balanced, X_test_balanced, y_train, y_val_balanced, y_test_balanced


# Helper function to balance dataset by undersampling larger classes
def balance_dataset(X, y, balance_percentage=1.0):
    class_counts = Counter(y)
    min_class_size = int(min(class_counts.values()) * balance_percentage)  # Adjust class size based on balance_percentage
    
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


from multiprocessing import Pool
from functools import partial
import numpy as np

def apply_mixup(X, y, alpha=0.2, factor=1.5, batch_size=32, num_workers=4):
    # Number of additional samples needed
    target_size = int(len(X) * factor)
    additional_samples = target_size - len(X)
    
    # Split the dataset into chunks aligned with batch size for consistency
    def chunk_data(X, y, chunk_size):
        return [(X[i:i+chunk_size], y[i:i+chunk_size]) for i in range(0, len(X), chunk_size)]

    # Create a pool of workers
    with Pool(processes=num_workers) as pool:
        # Create a partial function to apply MixUp on each chunk
        apply_mixup_chunk = partial(apply_mixup_batch, alpha=alpha)

        # Chunk the data
        data_chunks = chunk_data(X, y, batch_size)

        # Apply MixUp in parallel on each chunk
        result = pool.starmap(apply_mixup_chunk, data_chunks)

    # Concatenate results and select only the required number of samples
    mixed_X = np.concatenate([r[0] for r in result], axis=0)[:additional_samples]
    mixed_y = np.concatenate([r[1] for r in result], axis=0)[:additional_samples]

    # Combine original and new samples
    X_mixed = np.concatenate([X, mixed_X], axis=0)
    y_mixed = np.concatenate([y, mixed_y], axis=0)

    return X_mixed, y_mixed

# Helper function to apply MixUp augmentation on a single batch
def apply_mixup_batch(X, y, alpha=0.2):
    batch_size = len(X)
    
    # Generate lambda values using the Beta distribution
    lambda_vals = np.random.beta(alpha, alpha, batch_size)
    
    # Generate a permutation of indices for shuffling
    indices = np.random.permutation(batch_size)
    
    # Shuffle images and labels together
    shuffled_X = X[indices]
    shuffled_y = y[indices]
    
    # Apply MixUp
    mixed_X = lambda_vals[:, None, None, None] * X + (1 - lambda_vals[:, None, None, None]) * shuffled_X
    mixed_y = lambda_vals[:, None] * y + (1 - lambda_vals[:, None]) * shuffled_y
    
    return mixed_X, mixed_y


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
