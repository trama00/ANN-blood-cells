import tensorflow as tf

class PreprocessLayer(tf.keras.layers.Layer):
    """
    A custom Keras layer that preprocesses images by rescaling and removing backgrounds.
    
    This layer performs two main operations:
    1. Rescales images from [0, 255] to [0, 1] range if necessary
    2. Removes lighter backgrounds using a threshold-based approach
    """
    
    def __init__(self, threshold=0.5, name=None, **kwargs):
        """
        Initialize the preprocessing layer.
        
        Args:
            threshold (float): Intensity threshold to separate dark (subject) 
                             and light (background) regions. Default is 0.5.
        """
        super(PreprocessLayer, self).__init__(name=name, **kwargs)
        self.threshold = tf.Variable(threshold, trainable=False, dtype=tf.float32)
    
    @tf.function
    def call(self, inputs):
        """
        Process the input images.
        
        Args:
            inputs: Input tensor of shape [batch_size, height, width, channels]
        
        Returns:
            Processed images with backgrounds removed
        """
        # Input validation
        if not isinstance(inputs, tf.Tensor):
            inputs = tf.convert_to_tensor(inputs)
        
        if inputs.dtype != tf.float32:
            inputs = tf.cast(inputs, tf.float32)
        
        # Check if the image is within the [0, 255] range and rescale if necessary
        def check_and_rescale(image):
            max_val = tf.reduce_max(image)
            return tf.cond(
                max_val > 1.0,
                lambda: image / 255.0,
                lambda: image
            )
        
        # Apply check and rescale before background removal
        images_rescaled = tf.map_fn(check_and_rescale, inputs)
        
        # Apply background removal
        images_processed = tf.map_fn(
            lambda x: self.remove_background(x, self.threshold),
            images_rescaled
        )
        
        return images_processed
    
    @tf.function
    def remove_background(self, image, threshold=0.5):
        """
        Removes the background by masking out lighter pixels.
        Assumes that the subject (darker) is below the threshold and the background (lighter) is above.

        Args:
            image: Input image tensor of shape [height, width, channels]
            threshold: Intensity threshold to separate dark (subject) and light (background) regions

        Returns:
            The image with the background removed
        """
        # Create a mask where darker areas are 1 (subject)
        mask = tf.where(image < threshold, 1.0, 0.0)
        
        # For RGB images, we want the same mask applied to all channels
        if len(image.shape) == 3 and image.shape[-1] > 1:
            # Use reduce_mean to create a single channel mask
            mask = tf.reduce_mean(mask, axis=-1, keepdims=True)
            mask = tf.broadcast_to(mask, tf.shape(image))
        
        return image * mask
    
    def get_config(self):
        """
        Returns the layer configuration for serialization.
        """
        config = super().get_config()
        config.update({
            "threshold": float(self.threshold.numpy())
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """
        Creates a layer from its config.
        """
        return cls(**config)
