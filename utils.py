import numpy as np
import tensorflow as tf
from multiprocessing import Pool
from tensorflow.keras.applications import (
    Xception, VGG16, VGG19, ResNet50, ResNet50V2, ResNet101, ResNet101V2,
    ResNet152, ResNet152V2, InceptionV3, InceptionResNetV2, MobileNet,
    MobileNetV2, DenseNet121, DenseNet169, DenseNet201, NASNetMobile,
    NASNetLarge, EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,
    EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7,
    EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2, EfficientNetV2B3,
    EfficientNetV2S, EfficientNetV2M, EfficientNetV2L, ConvNeXtTiny,
    ConvNeXtSmall, ConvNeXtBase, ConvNeXtLarge, ConvNeXtXLarge
)
# Function to check if an image matches the conditions
def check_image(i, img, shrek, troll, tol):
    """Check if an image matches the shrek or troll images within a tolerance."""
    if (np.allclose(img, shrek, atol=tol) or np.allclose(img, troll, atol=tol)) and i != 11959 and i != 13559:
        return i
    return None


# Function to parallelize the process of finding the indices to remove
def parallel_index_removal(data, shrek, troll, tol=0.0001, num_workers=4):
    """Parallelize the removal of images that match 'shrek' or 'troll'."""
    with Pool(processes=num_workers) as pool:
        results = pool.starmap(check_image, [(i, img, shrek, troll, tol) for i, img in enumerate(data)])

    # Filter out None values (no matches) and return the indices to remove
    index_to_remove = [index for index in results if index is not None]
    
    return index_to_remove



def get_base_model(model_name, input_shape):
    models = {
        'Xception': Xception, 'VGG16': VGG16, 'VGG19': VGG19, 'ResNet50': ResNet50,
        'ResNet50V2': ResNet50V2, 'ResNet101': ResNet101, 'ResNet101V2': ResNet101V2,
        'ResNet152': ResNet152, 'ResNet152V2': ResNet152V2, 'InceptionV3': InceptionV3,
        'InceptionResNetV2': InceptionResNetV2, 'MobileNet': MobileNet, 'MobileNetV2': MobileNetV2,
        'DenseNet121': DenseNet121, 'DenseNet169': DenseNet169, 'DenseNet201': DenseNet201,
        'NASNetMobile': NASNetMobile, 'NASNetLarge': NASNetLarge, 'EfficientNetB0': EfficientNetB0,
        'EfficientNetB1': EfficientNetB1, 'EfficientNetB2': EfficientNetB2, 'EfficientNetB3': EfficientNetB3,
        'EfficientNetB4': EfficientNetB4, 'EfficientNetB5': EfficientNetB5, 'EfficientNetB6': EfficientNetB6,
        'EfficientNetB7': EfficientNetB7, 'EfficientNetV2B0': EfficientNetV2B0, 'EfficientNetV2B1': EfficientNetV2B1,
        'EfficientNetV2B2': EfficientNetV2B2, 'EfficientNetV2B3': EfficientNetV2B3, 'EfficientNetV2S': EfficientNetV2S,
        'EfficientNetV2M': EfficientNetV2M, 'EfficientNetV2L': EfficientNetV2L, 'ConvNeXtTiny': ConvNeXtTiny,
        'ConvNeXtSmall': ConvNeXtSmall, 'ConvNeXtBase': ConvNeXtBase, 'ConvNeXtLarge': ConvNeXtLarge,
        'ConvNeXtXLarge': ConvNeXtXLarge
    }
    
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' is not available. Choose from: {list(models.keys())}")
    
    base_model = models[model_name](include_top=False, input_shape=input_shape, weights='imagenet')
    base_model.trainable = False  # Freeze the base model

    return base_model