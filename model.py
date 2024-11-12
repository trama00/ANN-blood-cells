import os
import tensorflow as tf
import numpy as np

class PreprocessLayer(tf.keras.layers.Layer):
    def __init__(self, threshold=0.5, name=None, **kwargs):
        super(PreprocessLayer, self).__init__(name=name, **kwargs)
        self.threshold = tf.Variable(threshold, trainable=False, dtype=tf.float32)
    
    @tf.function
    def call(self, inputs):
        if not isinstance(inputs, tf.Tensor):
            inputs = tf.convert_to_tensor(inputs)
        
        if inputs.dtype != tf.float32:
            inputs = tf.cast(inputs, tf.float32)
            
        def check_and_rescale(image):
            max_val = tf.reduce_max(image)
            return tf.cond(
                max_val > 1.0,
                lambda: image / 255.0,
                lambda: image
            )
        
        images_rescaled = tf.map_fn(check_and_rescale, inputs)
        images_processed = tf.map_fn(
            lambda x: self.remove_background(x, self.threshold),
            images_rescaled
        )
        return images_processed
    
    @tf.function
    def remove_background(self, image, threshold=0.5):
        mask = tf.where(image < threshold, 1.0, 0.0)
        if len(image.shape) == 3 and image.shape[-1] > 1:
            mask = tf.reduce_mean(mask, axis=-1, keepdims=True)
            mask = tf.broadcast_to(mask, tf.shape(image))
        return image * mask
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "threshold": float(self.threshold.numpy())
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    
class ConditionalAugmentation(tf.keras.layers.Layer):
    
    
    def __init__(self, augmentation_layers, name=None, **kwargs):
        
        super(ConditionalAugmentation, self).__init__(name=name, **kwargs)
        self.augmentation_layers = augmentation_layers
        
    def build(self, input_shape):
        
        super().build(input_shape)

    def call(self, inputs, training=False, **kwargs):
        
        if training:
            return self.augmentation_layers(inputs)
        return inputs
    
    def get_config(self):
        
        config = super().get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        
        return cls(**config)
    

class Model:
    def __init__(self):
        """Initializes the model by finding and loading the .keras file."""
        # Look for a file with a .keras extension in the current directory
        keras_files = [f for f in os.listdir('.') if f.endswith('.keras')]
        
        if len(keras_files) == 1:
            self.model = tf.keras.models.load_model(
                keras_files[0],
                custom_objects={
                    'PreprocessLayer': PreprocessLayer,
                    'ConditionalAugmentation': ConditionalAugmentation
                }
            )            
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