import tensorflow as tf
import keras_cv as kcv

# -------------------------------------------- #
# Custom Function for White Noise Augmentation
# -------------------------------------------- #

# Function to add white noise to images
def add_white_noise(images, noise_stddev=10.0):
    """
    Adds white Gaussian noise to a batch of images without altering their pixel range (0 to 255).

    Args:
        images (tf.Tensor): Batch of images with shape [batch_size, height, width, channels] and pixel values in [0, 255].
        noise_stddev (float): Standard deviation of the Gaussian noise.

    Returns:
        tf.Tensor: Noisy images clipped to the range [0, 255].
    """
    # Generate white noise
    noise = tf.random.normal(shape=tf.shape(images), mean=0.0, stddev=noise_stddev, dtype=tf.float32)
    
    # Add noise to images
    noisy_images = images + noise
    
    # Clip pixel values to maintain valid range
    noisy_images = tf.clip_by_value(noisy_images, 0.0, 255.0)
    
    return noisy_images


# -------------------- #
# Keras CV Augmentation Layers
# -------------------- #

SEED = 2024  # Random seed for reproducibility

# Light Augmentations - Color Adjustments
random_saturation = kcv.layers.RandomSaturation(
    factor=(0.1, 0.9), seed=SEED
)
auto_contrast = kcv.layers.AutoContrast(
    value_range=(0, 255)
)
random_channel_shift = kcv.layers.RandomChannelShift(
    value_range=(0, 255), factor=(0.1, 0.9), channels=3, seed=SEED
)

# Light Augmentations - Distortions
random_sharpness = kcv.layers.RandomSharpness(
    factor=(0.1, 0.9), value_range=(0, 255), seed=SEED
)

# Medium Augmentations - Color Transformations
solarization = kcv.layers.Solarization(
    value_range=(0, 255), seed=SEED
)
random_hue = kcv.layers.RandomHue(
    factor=(0.1, 0.9), value_range=(0, 255), seed=SEED
)
random_color_degeneration = kcv.layers.RandomColorDegeneration(
    factor=(0.4, 0.8), seed=SEED
)

# Heavy Augmentations - Spatial Cuts
grid_mask = kcv.layers.GridMask(
    ratio_factor=(0.1, 0.5), rotation_factor=0.2,
    fill_mode="constant", fill_value=0.0, seed=SEED
)
random_cutout = kcv.layers.RandomCutout(
    height_factor=0.2, width_factor=0.2,
    fill_mode="constant", fill_value=0.0, seed=SEED
)

# Heavy Augmentations - Distortions
augmix = kcv.layers.AugMix(
    value_range=(0, 255), severity=(0.25, 0.75),
    num_chains=4, chain_depth=[2, 5], alpha=1.0, seed=SEED
)
random_shear = kcv.layers.RandomShear(
    x_factor=0.2, y_factor=0.2,
    interpolation="bilinear", fill_mode="constant",
    fill_value=0.0, seed=SEED
)
jittered_resize = kcv.layers.JitteredResize(
    target_size=(96, 96), scale_factor=(0.8, 1.2),
    crop_size=(64, 64), interpolation="bilinear", seed=SEED
)
rand_augment = kcv.layers.RandAugment(
    value_range=(0, 255), augmentations_per_image=4,
    magnitude=0.6, magnitude_stddev=0.2, rate=0.8
)

# Mix-Based Augmentations 
cutmix = kcv.layers.CutMix(
    alpha=1.0, seed=SEED
)
fourier_mix = kcv.layers.FourierMix(
    alpha=0.5, decay_power=3, seed=SEED
)
mixup = kcv.layers.MixUp(
    alpha=0.2, seed=SEED
)

# -------------------- #
# Augmentation Pipeline Settings
# -------------------- #

# Number of augmentations to select from each category
augpi_light = 4
augpi_medium = 4
augpi_distortion = 2
augpi_custom = 2
augpi_mix = 3

# Probability of applying each augmentation category
RATE_light = 0.8        # Light augmentations
RATE_medium = 0.8       # Medium-strength augmentations
RATE_distortion = 0.9   # Distortion augmentations
RATE_custom = 0.9       # Custom augmentations
RATE_mix = 0.9          # Mix-based augmentations

# Define augmentation layers for each pipeline
layers_light = [random_saturation, auto_contrast, random_channel_shift, random_sharpness]
layers_medium = [solarization, random_hue, random_color_degeneration]
layers_distortion = [random_shear, jittered_resize]
layers_custom = [augmix, random_cutout]
layers_mix = [cutmix, fourier_mix, mixup]

# -------------------- #
# Define Augmentation Pipelines
# -------------------- #

# Pipeline for light augmentations
pipeline_light = kcv.layers.RandomAugmentationPipeline(
    layers=layers_light,
    num_layers=augpi_light,
    rate=RATE_light,
    auto_vectorize=False,
    seed=SEED
)  # Applies light color adjustments

# Pipeline for medium-strength augmentations
pipeline_medium = kcv.layers.RandomAugmentationPipeline(
    layers=layers_medium,
    num_layers=augpi_medium,
    rate=RATE_medium,
    auto_vectorize=False,
    seed=SEED
)  # Applies medium color transformations

# Pipeline for distortion augmentations
pipeline_distortion = kcv.layers.RandomAugmentationPipeline(
    layers=layers_distortion,
    num_layers=augpi_distortion,
    rate=RATE_distortion,
    auto_vectorize=False,
    seed=SEED
)  # Applies spatial distortions

# Pipeline for customized augmentations
pipeline_custom = kcv.layers.RandomAugmentationPipeline(
    layers=layers_custom,
    num_layers=augpi_custom,
    rate=RATE_custom,
    auto_vectorize=False,
    seed=SEED
)  # Applies additional augmentations like AugMix and Cutout

# Pipeline for mix-based augmentations
pipeline_mix = kcv.layers.RandomAugmentationPipeline(
    layers=layers_mix,
    num_layers=augpi_mix,
    rate=RATE_mix,
    auto_vectorize=False,
    seed=SEED
)  # Combines images using mix-based techniques like CutMix

# Example of integrating pipelines into a model or data preprocessing step
# augmented_images = pipeline_light(images)
# augmented_images = pipeline_custom(augmented_images)
