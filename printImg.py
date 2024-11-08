import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

# Load dataset
dataset = np.load("data/training_set.npz", allow_pickle=True)

# Separate data and labels
data = dataset['images']
labels = dataset['labels']

def process_batch(batch_index, data, labels, N, M, images_per_pdf, folder):
    """
    Process a single batch of images, generate a PDF, and save it.
    """
    total_images = data.shape[0]
    i = batch_index * images_per_pdf  # Calculate the start index for this batch
    
    fig, axs = plt.subplots(N, M, figsize=(12, 10))
    fig.suptitle(f'Batch {batch_index + 1}')
    
    # Disable axes for all subplots
    for ax in axs.flat:
        ax.axis('off')
    
    # Plot images in the N x M grid
    for idx in range(images_per_pdf):
        if i + idx < total_images:
            img = data[i + idx] / 255.0
            label = labels[i + idx]
            row, col = divmod(idx, M)
            
            # Display image and add label and index to the corners
            axs[row, col].imshow(img)
            
            # Add label to the bottom-right corner
            axs[row, col].text(
                img.shape[1] - 10, img.shape[0] - 10,  # Position near the bottom-right corner
                label,
                color="white",
                fontsize=12,
                ha='right',
                va='bottom',
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5)
            )

            # Add index to the top-left corner
            axs[row, col].text(
                10, 10,  # Position near the top-left corner
                str(i + idx),  # The index of the image
                color="white",
                fontsize=12,
                ha='left',
                va='top',
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5)
            )
    
    # Save the current batch as a PDF
    pdf_path = os.path.join(folder, f'batch_{batch_index + 1}.pdf')
    plt.savefig(pdf_path, format='pdf')
    plt.close(fig)

def save_images_to_pdf_parallel(data, labels, N=4, M=5, folder="imgPrint", debug=False, num_workers=4):
    """
    Save images in batches of N x M as PDFs in the specified folder, in parallel.
    """
    # Ensure the output folder exists
    os.makedirs(folder, exist_ok=True)

    # Determine the total number of images and images per PDF
    total_images = data.shape[0]
    images_per_pdf = N * M
    total_batches = (total_images + images_per_pdf - 1) // images_per_pdf
    
    # If debug mode is enabled, only process the first batch
    if debug:
        total_batches = 1
    
    # Create a Pool of workers and process the batches in parallel
    with Pool(processes=num_workers) as pool:
        pool.starmap(process_batch, [(batch_index, data, labels, N, M, images_per_pdf, folder) 
                                     for batch_index in range(total_batches)])
    
    print(f"Saved {total_batches} PDF files in '{folder}'.")

# Example usage:
save_images_to_pdf_parallel(data, labels, N=8, M=10, debug=False)
