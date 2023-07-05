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
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
import os
import warnings
import tempfile
import urllib.request
from urllib.request import urlopen
import json
import path
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

fashion_mnist = keras.datasets.fashion_mnist


warnings.filterwarnings("ignore")
model = whisper.load_model("base.en")

if __name__ == '__main__':
    pass


# Set a "base" URL to append onto
base_url = "https://www.grailed.com/designers/20471120"
COOKIES_PATH = "cookies.pkl"
time1 = random.randint(2, 6)
MP3_PATH = r'C:\Users\Jake Gurien\PycharmProjects\mcjpbot'

def login_to_grailed(username, password):
    # Instantiate the WebDriver (e.g., Chrome driver)
    driver = webdriver.Chrome()
    login_url = "https://www.grailed.com/users/sign_up"
    driver.get(login_url)
    wait = WebDriverWait(driver, 20)
    try:
        element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.LINK_TEXT, "Log in"))
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
    #pickle.dump(driver.get_cookies(), open(COOKIES_PATH, "wb"))
    navigate_to_brand(driver, " ")
    return 0


def navigate_to_brand(driver, brand):
    search_bar = driver.find_element(By.CSS_SELECTOR, "input#header_search-input")
    search_bar.clear()
    search_bar.send_keys(brand)
    search_bar.send_keys(Keys.RETURN)
    time.sleep(time1)
    auto_scroll(driver)
    return 0


def auto_scroll(driver):
    results = driver.find_elements(By.XPATH, '//div[@class="FiltersInstantSearch"]//div[@class="feed-item"]')
    len(results)
    ListingNumber = driver.find_element(By.XPATH,
        '//div[@class="FiltersInstantSearch"]//div[@class="-header"]/span').text
    ListingNumber = int(ListingNumber.split(" ")[0].replace(",", ""))
    ScrollNumber = round(ListingNumber / 2000)
    last_height = driver.execute_script("return document.body.scrollHeight")
    for i in range(0, ScrollNumber):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
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
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    titles1 = soup.find_all('p', {'data-cy': 'listing-title', 'class': 'ListingMetadata-module__title___Rsj55'})
    for title in titles1:
        titles.append(title.text)
    
 
titles = []
username = "dazzlesdaddy@gmail.com"
password = "Jakey050603#"
login_to_grailed(username, password)
data = {
    'Title': titles
}
df = pd.DataFrame(data)
current_dir = os.getcwd()
file_path = os.path.join(current_dir, 'filename.csv')
df.to_csv(file_path, index=False)
