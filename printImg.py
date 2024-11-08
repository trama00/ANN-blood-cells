# imports
import os
import numpy as np
import matplotlib.pyplot as plt

# load dataset
dataset = np.load("data/training_set.npz", allow_pickle=True)

# separate data and labels
data = dataset['images']
labels = dataset['labels']

def save_images_to_pdf(data, labels, N=4, M=5, folder="imgPrint", debug=False):
    """
    Save images in batches of N x M as PDFs in the specified folder, with labels and indices in the corners.

    Parameters:
    - data: numpy array of images with shape (num_images, height, width, channels).
    - labels: ndarray with the same length as data, containing "healthy" or "unhealthy" labels.
    - N, M: integers specifying the grid layout (e.g., 4x5).
    - folder: name of the output folder where PDFs will be saved.
    - debug: boolean, if True, only the first N x M batch will be saved for testing.
    """
    # Ensure the output folder exists
    os.makedirs(folder, exist_ok=True)

    # Determine the total number of images and images per PDF
    total_images = data.shape[0]
    images_per_pdf = N * M
    pdf_counter = 1

    # Loop over the dataset in chunks
    for i in range(0, total_images, images_per_pdf):
        fig, axs = plt.subplots(N, M, figsize=(12, 10))
        fig.suptitle(f'Batch {pdf_counter}')
        
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
        pdf_path = os.path.join(folder, f'batch_{pdf_counter}.pdf')
        plt.savefig(pdf_path, format='pdf')
        plt.close(fig)
        pdf_counter += 1

        # If debug mode is enabled, only process the first batch and stop
        if debug:
            break

    print(f"Saved {pdf_counter - 1} PDF files in '{folder}'.")

# Example usage:
save_images_to_pdf(data, labels, N=8, M=10, debug=False)
