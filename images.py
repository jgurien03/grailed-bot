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
import seaborn as sns
import matplotlib.colors as mcolors
from datetime import datetime, timedelta, date
from selenium.webdriver.support.ui import Select
from matplotlib.cm import ScalarMappable
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
import cv2
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import NoSuchElementException
from keras.losses import CategoricalCrossentropy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dropout, BatchNormalization
from gensim.models import Word2Vec
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import imagehash
import nltk
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
from nltk.corpus import wordnet
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
from keras.applications.xception import preprocess_input
from keras.applications.xception import decode_predictions
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.applications.resnet import ResNet50, preprocess_input, decode_predictions
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Input
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
import matplotlib.cm as cm
from gensim.models import LdaModel
from gensim.corpora import Dictionary

fashion_mnist = keras.datasets.fashion_mnist


warnings.filterwarnings("ignore")
model = whisper.load_model("base.en")
nltk.download('wordnet')

if __name__ == '__main__':
    pass


# Set a "base" URL to append onto
base_url = "https://www.grailed.com/designers/20471120"
COOKIES_PATH = "cookies.pkl"
time1 = random.randint(2, 6)
MP3_PATH = r'{}'.format(os.getcwd())

def login_to_grailed(username, password):
    # Instantiate the WebDriver (e.g., Chrome driver)
    service = Service(executable_path=r'/usr/bin/chromedriver')
    options = webdriver.ChromeOptions()
    #options.add_argument('--window-size=1920,1080')
    #options.add_argument('--headless')
    driver = webdriver.Chrome(service=service, options=options)
    login_url = "https://www.grailed.com/users/sign_up"
    driver.get(login_url)
    wait = WebDriverWait(driver, 20)
    try:
        element = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//a[@href="/users/sign_up"]'))
        )
        time.sleep(2)
        element.click()
        print("success in clicking Login in button")
    except:
        print("Bot could not click on login button.")
        driver.quit()
    try:
        time.sleep(2)
        driver.find_element(By.CSS_SELECTOR, "button[data-cy='login-with-email']").click()
        print("success in clicking Login in button")
    except:
        print("Bot could not click on login button.")
        driver.quit()
    wait.until(EC.element_to_be_clickable((By.ID, "email"))).send_keys(username)
    time.sleep(2)
    wait.until(EC.element_to_be_clickable((By.ID, "password"))).send_keys(password)
    try:
        element = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-cy='auth-login-submit']"))
        )
        time.sleep(2)
        element.click()
        print("success!")
    except:
        driver.quit()
        print("failed!")
    try:
        submit_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit']"))
        )
        time.sleep(2)
        submit_button.click()
    except:
        if driver.current_url == 'https://www.grailed.com/':
            print("success!")
        else:
            try:
                time.sleep(2)
                click_button(driver)
                print('success!!!!')
                time.sleep(2)
                driver.find_element(By.CLASS_NAME, "rc-audiochallenge-play-button").click()
                time.sleep(5)
                play_final(driver)
                time.sleep(5)
            except:
                print('failed!')
                driver.quit()
    navigate_to_brand(driver, brand)
    return 0


def navigate_to_brand(driver, brand):
    search_bar = driver.find_element(By.CSS_SELECTOR, "input#header_search-input")
    search_bar.clear()
    search_bar.send_keys(brand)
    search_bar.send_keys(Keys.RETURN)
    time.sleep(time1)
    if response.lower() == 'sold':
        try:
            WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//button/span[contains(text(), 'Filter')]"))).click()
        except TimeoutException:
            window_width = 800
            window_height = driver.get_window_size()['height']
            driver.set_window_size(window_width, window_height)
            WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//button/span[contains(text(), 'Filter')]"))).click()
        button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//button[contains(@class, '-attribute-item') and span[contains(@class, '-attribute-header') and text()='Show Only']]"))).click()        
        checkbox = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input.-toggle[name='sold']"))).click()
        time.sleep(1)
        auto_scroll(driver)
    else:
        auto_scroll(driver)
    return 0


def auto_scroll(driver):
    results = driver.find_elements(By.XPATH, '//div[@class="FiltersInstantSearch"]//div[@class="feed-item"]')
    len(results)
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.5)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    results = driver.find_elements(By.XPATH, '//div[@class="FiltersInstantSearch"]//div[@class="feed-item"]')
    sort_results(driver, results)


def click_button(driver):
    driver.switch_to.default_content()
    driver.switch_to.frame(driver.find_element(By.XPATH, ".//iframe[@title='recaptcha challenge expires in two minutes']"))
    driver.find_element(By.ID, "recaptcha-audio-button").click()


def transcribe1(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        filename = os.path.join(save_path, 'file.mp3')
        with open(filename, 'wb') as f:
            f.write(requests.get(url).content)
        print('Download complete.')
        assert os.path.isfile(filename)
        with open(filename, "r") as f:
            pass
        result = model.transcribe('file.mp3')
        return result["text"].strip()
    else:
        print('Failed to download the file.')
        return None


def play_final(driver):
    text = transcribe1(driver.find_element(By.ID, "audio-source").get_attribute('src'), MP3_PATH)
    driver.find_element(By.ID, "audio-response").send_keys(text)
    driver.find_element(By.ID, "recaptcha-verify-button").click()


def sort_results(driver, results):
    for WebElement in results:
        elementHTML = WebElement.get_attribute('outerHTML')
        element_soup = BeautifulSoup(elementHTML,'html.parser')
        image = element_soup.find('img')
        if image:
            image_url = image['src']
        listing_images.append(image_url)

def download_images(image_links, save_dir):
    os.makedirs(save_dir, exist_ok=True)  
    for i, link in enumerate(image_links):
        image_name = f"image_{i}.jpg"
        save_path = os.path.join(save_dir, image_name) 
        try:
            urllib.request.urlretrieve(link, save_path)
            print(f"Downloaded {image_name}")
        except Exception as e:
            print(f"Error downloading {image_name}: {str(e)}")

def find_duplicate_images(folder_path):
    hash_dict = {}
    duplicate_images = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isdir(file_path) or not is_image_file(file_path):
            continue
        image = Image.open(file_path)
        image_hash = imagehash.average_hash(image)
        if image_hash in hash_dict:
            duplicate_images.append((file_path, hash_dict[image_hash]))
        else:
            hash_dict[image_hash] = file_path
    return duplicate_images

def is_image_file(file_path):
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
    return any(file_path.lower().endswith(ext) for ext in image_extensions)

def delete_duplicate_images(duplicate_images):
    for file_path, original_file_path in duplicate_images:
        print(f"Deleting duplicate: {file_path}")
        os.remove(file_path)

listing_images = []
save_dir = r"C:\Users\jguri\Desktop\Grailed-Scraper\new_dataset\validation\Tops"
username = "dazzlesdaddy@gmail.com"
password = "Jakey050603#"
brand = input("What brand would you like to search? ")
response = input("Would you like to look at current or sold listings? (current/sold) ")
login_to_grailed(username, password)
download_images(listing_images, save_dir)
duplicate_images = find_duplicate_images(save_dir)
delete_duplicate_images(duplicate_images)