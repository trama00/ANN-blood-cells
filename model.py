import os
import tensorflow as tf
import numpy as np

class Model:
    def __init__(self):
        """Initializes the model by finding and loading the .keras file."""
        # Look for a file with a .keras extension in the current directory
        keras_files = [f for f in os.listdir('.') if f.endswith('.keras')]
        
        if len(keras_files) == 1:
            self.model = tf.keras.models.load_model(keras_files[0])
            print(f"Loaded model from {keras_files[0]}")
        elif len(keras_files) == 0:
            raise FileNotFoundError("No .keras file found in the current directory.")
        else:
            raise ValueError("Multiple .keras files found. Ensure only one .keras file is present.")

    def predict(self, X):
        """Returns a numpy array of labels for the given input X."""
        
        # Use `predict` to get the predictions
        predictions = self.model.predict(X)
        
        # Get the predicted class (highest probability)
        predicted_classes = np.argmax(predictions, axis=1)
        return predicted_classes