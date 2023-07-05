import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import time
import csv
import string
import sys
import random
import requests
import datefinder
import pickle
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
import cv2
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from keras.losses import CategoricalCrossentropy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dropout, BatchNormalization
from gensim.models import Word2Vec
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import bs4
from sklearn.neighbors import KNeighborsClassifier
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
import os
import warnings
import tempfile
import urllib.request
from urllib.request import urlopen
import json
import path
from sklearn.cluster import KMeans
from PIL import Image, ImageOps
from io import BytesIO
import whisper
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from playsound import playsound, PlaysoundException
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from keras.layers import Dense, Flatten
from keras.models import Sequential, load_model
import keras.utils as image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.xception import preprocess_input
from keras.applications.xception import decode_predictions
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.applications.resnet import ResNet50, preprocess_input, decode_predictions
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Input
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from gensim.models import LdaModel
from gensim.corpora import Dictionary

def make_model(learning_rate, num_classes):
    base_model = Xception(
        weights='imagenet',
        input_shape=(150, 150, 3),
        include_top=False
    )
    base_model.trainable = False
    inputs = Input(shape=(150, 150, 3))
    base = base_model(inputs, training=False)
    vector = GlobalAveragePooling2D()(base)
    outputs = Dense(num_classes, activation='softmax')(vector)
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
def train_model(train_dir, val_dir, learning_rate=0.001, epochs=18, batch_size=32, save_path='model.h5'):
    image_size = (150, 150)
    train_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.xception.preprocess_input)
    train_ds = train_gen.flow_from_directory(
        train_dir,
        seed=1,
        target_size=image_size,
        batch_size=batch_size,
    )
    validation_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.xception.preprocess_input)
    val_ds = validation_gen.flow_from_directory(
        val_dir,
        seed=1,
        target_size=image_size,
        batch_size=batch_size,
    )
    num_classes = len(train_ds.class_indices)
    model = make_model(learning_rate, num_classes)
    early_stopping = EarlyStopping(patience=2, restore_best_weights=True)
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[early_stopping]
    )
    model.save(save_path)
    return model, history

train_dir = 'new_dataset/train'
val_dir = 'new_dataset/validation'
model, history = train_model(train_dir, val_dir, learning_rate=0.001, epochs=18, batch_size=32, save_path='model.h5')