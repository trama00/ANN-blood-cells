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


class ConditionalAugmentation(tf.keras.layers.Layer):
    """
    A custom Keras layer that conditionally applies data augmentation based on 
    the training mode. This layer enables augmentation to be used only during 
    training and bypassed during inference.

    Attributes:
    ----------
    augmentation_layers : tf.keras.Sequential
        A sequential model or list of augmentation layers that define the 
        augmentation pipeline to be applied conditionally.

    Methods:
    -------
    call(inputs, labels=None, training=False, **kwargs)
        Conditionally applies the augmentation layers when training is True; 
        otherwise, returns the input as-is.
    """
    
    def __init__(self, augmentation_layers, name=None, **kwargs):
        """
        Initializes the ConditionalAugmentation layer.

        Parameters:
        ----------
        augmentation_layers : tf.keras.Sequential
            The augmentation layers (e.g., RandomFlip, RandomRotation) to apply 
            conditionally during training.
        
        name : str, optional
            Optional name for the layer.
        
        **kwargs : 
            Additional keyword arguments for the Keras Layer superclass.
        """
        super(ConditionalAugmentation, self).__init__(name=name, **kwargs)
        self.augmentation_layers = augmentation_layers
        
    def build(self, input_shape):
        """
        Builds the layer based on the input shape.

        Since ConditionalAugmentation only wraps a Sequential model of augmentation layers,
        no additional actions are required during building. This method is included to
        suppress warnings from Keras, which expects a build method for all custom layers.
        
        Args:
            input_shape (tuple): Shape of the input data.
        """
        super().build(input_shape)

    def call(self, inputs, training=False, **kwargs):
        """
        Applies the augmentation layers only if training is True.
        
        Parameters:
        ----------
        images : Tensor
            Input image tensor to apply augmentation to if in training mode.
        
        labels : Tensor, optional
            Input label tensor, required for certain augmentations like MixUp.
        
        training : bool, optional
            Flag indicating whether the model is in training mode. If True, 
            applies the augmentation; if False, bypasses it.
        
        **kwargs : 
            Additional keyword arguments for the call method.
        
        Returns:
        -------
        Tensor or Tuple[Tensor, Tensor]
            The augmented image (and label if required) tensor if training is True,
            otherwise the original input tensor(s).
        """
        if training:
            return self.augmentation_layers(inputs)
        return inputs
    
    def get_config(self):
        """
        Returns the layer configuration for serialization.
        
        Returns:
            dict: Configuration dictionary containing initialization arguments.
        """
        config = super().get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        """
        Creates a layer from its config.

        Args:
            config (dict): Configuration dictionary.
        
        Returns:
            ConditionalAugmentation: A layer instance with the specified configuration.
        """
        return cls(**config)
