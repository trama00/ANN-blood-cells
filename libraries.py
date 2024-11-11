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
import keras_cv.layers as kcv_layers
from tensorflow.keras.applications import VGG16
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.preprocessing import label_binarize
from datetime import datetime
import cv2
from multiprocessing import Pool
import random
from sklearn.model_selection import train_test_split