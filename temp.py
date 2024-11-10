import numpy as np

# Load data
data = np.load('data/training_set.npz', allow_pickle=True)

# esplore dataset
print(data.files)
print(data['images'].shape)
print(data['labels'].shape)

