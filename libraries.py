# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from tensorflow import keras as tfk
from keras import layers as tfkl
# from keras_tuner import RandomSearch
# import keras_cv.layers as kcv_layers
# from tensorflow.keras.applications import EfficientNetB1
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
import cv2
from multiprocessing import Pool
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.convnext import preprocess_input
import re

from preprocess import one_hot_encode_labels, clean_dataset
from data_partitioning import split_and_balance_distribution, print_class_distribution, apply_mixup
from custom_layer import PreprocessLayer, ConditionalAugmentation
from utils import get_base_model, compute_class_weights, print_sample
from trainable_layers import set_trainable_layers