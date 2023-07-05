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
import torch
import torch.nn as nn
import io
from sklearn.preprocessing import MinMaxScaler
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
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
from selenium.common.exceptions import NoSuchElementException
from keras.losses import CategoricalCrossentropy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dropout, BatchNormalization
from gensim.models import Word2Vec
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
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
import locale
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
import discord
from discord.ext import commands
import asyncio

fashion_mnist = keras.datasets.fashion_mnist


warnings.filterwarnings("ignore")
model = whisper.load_model("base.en")
nltk.download('wordnet')

if __name__ == '__main__':
    pass

intents = discord.Intents.all()
intents.members = True
intents.typing = True
intents.presences = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Set a "base" URL to append onto
base_url = "https://www.grailed.com/designers/20471120"
COOKIES_PATH = "cookies.pkl"
time1 = random.randint(2, 6)
MP3_PATH = r'{}'.format(os.getcwd())

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name} ({bot.user.id})')


@bot.command()
async def search(ctx, username, password, response):
    await ctx.send("Please enter a brand:")
    brand_response = await bot.wait_for('message', check=lambda m: m.author == ctx.author and m.channel == ctx.channel, timeout=30)
    brand = brand_response.content.strip()
    await ctx.send(f"Searching for brand: {brand} with response: {response}")
    driver = create_webdriver()
    await asyncio.sleep(2)
    login_to_grailed(driver, username, password)
    await asyncio.sleep(2)
    navigate_to_brand(driver, brand, response)
    await ctx.send("Search complete.")

@bot.command()
async def analyze(ctx):
    await ctx.send("Please enter the brand (use the same brand that you searched earlier):")
    brand_response = await bot.wait_for('message', check=lambda m: m.author == ctx.author and m.channel == ctx.channel, timeout=30)
    brand = brand_response.content.strip()
    df = pd.DataFrame(data)
    df['Category'] = df['Title'].apply(clean_up_categories)
    df = filter_rows_by_keyword(df, brand)

    for index, row in df.iterrows():
        size = row['Size']
        if isinstance(size, str) and size.lower() == 'os':
            df.at[index, 'Category'] = 'Accessories'
        elif isinstance(size, (int, float)) and 22 <= size <= 50:
            df.at[index, 'Category'] = 'Bottoms'
        elif isinstance(size, (int, float)) and 4 <= size <= 15:
            df.at[index, 'Category'] = 'Shoes'

    for index, item in df.iterrows():
        if df.at[index, 'Category'] == 'Other':
            df.at[index, 'Category'] = predict_categories(model1, item['Image Link'])
            #await ctx.send(f"Predicted category for item {index}: {df.at[index, 'Category']}")

    df['Current Price'] = df['Current Price'].str.replace('[^\d.]', '', regex=True)
    df['Current Price'] = pd.to_numeric(df['Current Price'])
    df['Original Price'] = df['Original Price'].str.replace('[^\d.]', '', regex=True)
    df['Original Price'] = pd.to_numeric(df['Original Price'])
    df['Relative Date'] = df['Original Date'].apply(get_relative_date)

    await ctx.send("Would you like to see each category and their feed appearance count? (yes/no): ")
    response1 = await bot.wait_for('message', check=lambda m: m.author == ctx.author and m.channel == ctx.channel, timeout=30)
    if response1.content.lower() == 'yes':
        await visualize_category_distribution(ctx, df)
    else:
        await ctx.send("Graph display skipped.")
    await ctx.send("Would you like to see each category and their prices compared against each other? (yes/no): ")
    response2 = await bot.wait_for('message', check=lambda m: m.author == ctx.author and m.channel == ctx.channel, timeout=30)
    if response2.content.lower() == 'yes':
        await plot_category_prices(ctx, df, 'Category')
    else:
        await ctx.send("Graph display skipped.")

    await ctx.send("Would you like to see the brand's change in price over time? (yes/no): ")
    response3 = await bot.wait_for('message', check=lambda m: m.author == ctx.author and m.channel == ctx.channel, timeout=30)
    if response3.content.lower() == 'yes':
        df2 = df
        await ctx.send("Please enter a time unit (days, months, or years):")
        date_response = await bot.wait_for('message', check=lambda m: m.author == ctx.author and m.channel == ctx.channel, timeout=30)
        date = date_response.content.strip().lower()
        if date == 'days' or date == 'months' or date == 'years':
            await ctx.send("Would you like to visualize a certain category? (yes/no)")
            response4 = await bot.wait_for('message', check=lambda m: m.author == ctx.author and m.channel == ctx.channel, timeout=30)
            if response4.content.lower() == 'yes':
                await ctx.send("Select a category: (Tops, Bottoms, Skirts, Dresses, Shoes, Outerwear, Accessories)")
                response5 = await bot.wait_for('message', check=lambda m: m.author == ctx.author and m.channel == ctx.channel, timeout=30)
                df2 = filter_dataframe_by_category(df, 'Category', response5.content.strip())
                await plot_price_vs_upload_date(ctx, df2, date)
            else:
                await plot_price_vs_upload_date(ctx, df2, date)
        else:
            await ctx.send("Invalid time unit. Must specify days, months, or years.")
    else:
        await ctx.send("Graph display skipped.")

    await ctx.send("Would you like to see each size and their average price? (yes/no): ")
    response6 = await bot.wait_for('message', check=lambda m: m.author == ctx.author and m.channel == ctx.channel, timeout=30)
    if response6.content.lower() == 'yes':
        df3 = df
        await ctx.send("Would you like to visualize a certain category? (yes/no)")
        response7 = await bot.wait_for('message', check=lambda m: m.author == ctx.author and m.channel == ctx.channel, timeout=30)
        if response7.content.lower() == 'yes':
            await ctx.send("Select a category: (Tops, Bottoms, Skirts, Dresses, Shoes, Outerwear, Accessories)")
            response8 = await bot.wait_for('message', check=lambda m: m.author == ctx.author and m.channel == ctx.channel, timeout=30)
            df3 = filter_dataframe_by_category(df, 'Category', response8.content.strip())
            await plot_price_by_size(ctx, df3, 'Size', 'Original Price')
        else:
            await plot_price_by_size(ctx, df3, 'Size', 'Original Price')
    else:
        await ctx.send("Graph display skipped.")

    df4 = df
    await ctx.send("Would you like to see a prediction of the price in the future? (yes/no): ")
    response9 = await bot.wait_for('message', check=lambda m: m.author == ctx.author and m.channel == ctx.channel, timeout=30)
    if response9.content.lower() == 'yes':
        await ctx.send("Choose a month as a number between 1 and 12:")
        response10 = await bot.wait_for('message', check=lambda m: m.author == ctx.author and m.channel == ctx.channel, timeout=30)
        month = int(response10.content.strip())
        await ctx.send("Choose the current year:")
        response11 = await bot.wait_for('message', check=lambda m: m.author == ctx.author and m.channel == ctx.channel, timeout=30)
        year = int(response11.content.strip())
        predicted_price = predict_future_prices(df4, year, month)
        await ctx.send(f"Predicted price for {month}/{year}: {predicted_price}")
    else:
        await ctx.send("Prediction skipped.")

@bot.command()
@commands.is_owner()
async def shutdown(context):
    await context.send("Shutting down...")
    exit()

def create_webdriver():
    service = Service(executable_path=r'/usr/bin/chromedriver')
    options = Options()
    # options.add_argument('--headless')
    driver = webdriver.Chrome(service=service, options=options)
    return driver


def login_to_grailed(driver, username, password):
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


def navigate_to_brand(driver, brand, response):
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
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    titles1 = soup.find_all('p', {'data-cy': 'listing-title', 'class': 'ListingMetadata-module__title___Rsj55'})
    for title in titles1:
        titles.append(title.text)
    for WebElement in results:
        elementHTML = WebElement.get_attribute('outerHTML')
        element_soup = BeautifulSoup(elementHTML,'html.parser')
        price_element = element_soup.find(class_='ListingPriceAndHeart-module__listingPriceAndHeart___MEGdE')   
        if price_element:
            current_price_element = price_element.find(class_='Money-module__root___jRyq5', attrs={'data-testid': 'Current'})
            original_price_element = price_element.find(class_='Money-module__root___jRyq5 Price-module__original___I3r3D', attrs={'data-testid': 'Original'})      
            if current_price_element and original_price_element:
                item = current_price_element.text.strip()   
                old_item = original_price_element.text.strip()      
            elif current_price_element:
                item = current_price_element.text.strip()
                old_item = item
            prices.append(item)   
            old_prices.append(old_item) 
    for WebElement in results:
        elementHTML = WebElement.get_attribute('outerHTML')
        element_soup = BeautifulSoup(elementHTML,'html.parser')
        age_element = element_soup.find(class_='ListingAge-module__listingAge___EoWHC')
        if age_element:
            date_ago_element = age_element.find(class_='ListingAge-module__dateAgo___xmM8y')
            strike_through_element = age_element.find(class_='ListingAge-module__strikeThrough___LoORR')      
            if date_ago_element and strike_through_element:
                new_age = date_ago_element.text.strip()
                original_age = strike_through_element.text.strip()[1:-1]
                old_item = original_age         
            elif date_ago_element:
                new_age = date_ago_element.text.strip() 
                old_item = new_age  
            if '(' in new_age:
                index = new_age.index('(')
                new_age = new_age[:index]
            dates.append(new_age)
            old_dates.append(old_item)
    for WebElement in results:
        elementHTML = WebElement.get_attribute('outerHTML')
        element_soup = BeautifulSoup(elementHTML,'html.parser')
        image = element_soup.find('img')
        if image:
            image_url = image['src']
        listing_images.append(image_url)
    for link in soup.find_all('a', class_='listing-item-link'):
        href = link['href']
        listing_links.append(href)
    for WebElement in results:
        elementHTML = WebElement.get_attribute('outerHTML')
        element_soup = BeautifulSoup(elementHTML,'html.parser')
        size_element = element_soup.find('p', class_='ListingMetadata-module__size___e9naE')
        if size_element:
            size_text = size_element.get_text()
            sizes.append(size_text)

def predict_categories(model, image_paths):
    class_names = ['Accessories', 'Bottoms', 'Dresses', 'Outerwear', 'Shoes', 'Skirts', 'Tops']
    response = requests.get(image_paths)
    img = image.load_img(BytesIO(response.content), target_size=(150, 150))
    img = image.img_to_array(img)
    img = tf.expand_dims(img, axis=0)
    img = tf.keras.applications.xception.preprocess_input(img)
    predictions = model.predict(img)
    predicted_category = tf.argmax(predictions, axis=1)[0]
    return class_names[predicted_category]


def clean_up_categories(cell):
    cell_lower = cell.lower()
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in cell_lower:
                return category
            synonyms = wordnet.synsets(keyword)
            for synonym in synonyms:
                if synonym.lemmas()[0].name().lower() in cell_lower:
                    return category
    return 'Other'

async def visualize_category_distribution(ctx, df):
    category_counts = df['Category'].value_counts()
    plt.bar(category_counts.index, category_counts.values)
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title('Distribution of Categories')
    plt.xticks(rotation=45)
    filename = 'category_count.png'
    plt.savefig(filename)
    with open(filename, 'rb') as file:
        image = discord.File(file)
        await ctx.send(file=image)
    plt.close()

async def plot_category_prices(ctx, df, category_col):
    categories = df[category_col].unique()
    bar_positions = np.arange(len(categories))
    bar_width = 0.4
    category_prices = []
    for category in categories:
        category_prices.append(df[df[category_col] == category]['Current Price'].mean())
    plt.bar(bar_positions, category_prices, width=bar_width)
    plt.xticks(bar_positions, categories)
    plt.ylabel('Price')
    plt.title('Average Price Comparison ($)')
    filename = 'category_prices.png'
    plt.savefig(filename)
    with open(filename, 'rb') as file:
        image = discord.File(file)
        await ctx.send(file=image)
    plt.close()

def get_relative_date(time_string):
    current_datetime = datetime.now()
    if "Sold" in time_string:
        time_string = time_string.replace("Sold", "").strip()
    if "almost" in time_string:
        time_string = time_string.replace("almost", "").strip()
    if "over" in time_string:
        time_string = time_string.replace("over", "").strip()
    if "about" in time_string:
        time_string = time_string.replace("about", "").strip()
    if "minute" in time_string:
        minutes = int(time_string.split()[0])
        relative_date = current_datetime - timedelta(minutes=minutes)
    elif "hour" in time_string:
        hours = int(time_string.split()[0])
        relative_date = current_datetime - timedelta(hours=hours)
    elif "day" in time_string:
        days = int(time_string.split()[0])
        relative_date = current_datetime - timedelta(days=days)
    elif "month" in time_string:
        months = int(time_string.split()[0])
        relative_date = current_datetime - timedelta(days=months * 30)
    elif "year" in time_string:
        years = int(time_string.split()[0])
        relative_date = current_datetime - timedelta(days=years * 365)
    else:
        relative_date = current_datetime
    return relative_date


async def plot_price_vs_upload_date(ctx, df, date):
    if (date == 'days' and any(df['Original Date'].str.contains('day'))) or ((not any(df['Original Date'].str.contains('month')) and not any(df['Original Date'].str.contains('year'))) and any(df['Original Date'].str.contains('days'))):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=29)
        df['Day'] = df['Relative Date'].dt.day
        avg_prices = []
        volumes = []
        for day in range(1, 31):
            mask = (df['Day'] == day) & (df['Relative Date'] >= start_date) & (df['Relative Date'] <= end_date)
            day_data = df[mask]
            if not day_data.empty:
                avg_price = day_data['Original Price'].mean()
                volume = day_data['Title'].count()
                avg_prices.append(avg_price)
                volumes.append(volume)
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.set_xlabel("Day")
        ax1.set_ylabel("Average Price")
        ax1.plot(range(1, len(avg_prices) + 1), avg_prices, color='blue', label='Average Price')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_xticks(range(1, len(avg_prices) + 1))
        ax1.set_xticklabels([f"Day {day}" for day in range(1, len(avg_prices) + 1)], rotation=45)
        ax2 = ax1.twinx()
        ax2.set_ylabel("Volume", color='red')
        ax2.plot(range(1, len(volumes) + 1), volumes, color='red', label='Volume')
        ax2.tick_params(axis='y', labelcolor='red')
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')
        plt.tight_layout()
        filename = 'days.png'
        plt.savefig(filename)
        with open(filename, 'rb') as file:
            image = discord.File(file)
            await ctx.send(file=image)
        plt.close()
    elif date == 'months' and any(df['Original Date'].str.contains('month')):
        df_monthly_avg = df.groupby([df["Relative Date"].dt.year, df["Relative Date"].dt.month])["Original Price"].mean()
        df_monthly_avg = df_monthly_avg.sort_index(ascending=True)
        df_monthly_volume = df.groupby([df["Relative Date"].dt.year, df["Relative Date"].dt.month])["Title"].count()
        df_monthly_volume = df_monthly_volume.sort_index(ascending=True)
        min_volume = df_monthly_volume.min()
        max_volume = df_monthly_volume.max()
        normalized_volume = (df_monthly_volume - min_volume) / (max_volume - min_volume)
        x_labels = [f"{month}-{year}" for year, month in df_monthly_avg.index]
        x_ticks = np.arange(len(df_monthly_avg))
        cmap = plt.cm.get_cmap('viridis')
        colors = cmap(normalized_volume)
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(x_ticks, df_monthly_avg.values, color=colors)
        sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_volume, vmax=max_volume))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Volume')
        ax.set_xlabel("Month-Year")
        ax.set_ylabel("Average Price")
        ax.set_title("Monthly Average Price")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45)
        plt.tight_layout()
        filename = 'months.png'
        plt.savefig(filename)
        with open(filename, 'rb') as file:
            image = discord.File(file)
            await ctx.send(file=image)
        plt.close()
    elif date == 'years' and (any(df['Original Date'].str.contains('year')) or any(df['Original Date'].str.contains('years'))):
        df_yearly_avg = df.groupby(df["Relative Date"].dt.year)["Original Price"].mean()
        df_yearly_avg = df_yearly_avg.sort_index(ascending=True)
        df_yearly_volume = df.groupby(df["Relative Date"].dt.year)["Title"].count()
        df_yearly_volume = df_yearly_volume.sort_index(ascending=True)
        min_volume = df_yearly_volume.min()
        max_volume = df_yearly_volume.max()
        normalized_volume = (df_yearly_volume - min_volume) / (max_volume - min_volume)
        x_labels = df_yearly_avg.index
        x_ticks = np.arange(len(df_yearly_avg))
        cmap = plt.cm.get_cmap('viridis')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlabel("Year")
        ax.set_ylabel("Average Price")
        for i, (price, volume) in enumerate(zip(df_yearly_avg.values, normalized_volume)):
            color = cmap(volume)
            ax.bar(x_ticks[i], price, color=color, alpha=0.7, edgecolor='black')
        colorbar = plt.cm.ScalarMappable(cmap='viridis')
        colorbar.set_array(normalized_volume)
        plt.colorbar(colorbar, label='Volume')
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45)
        plt.tight_layout()
        filename = 'years.png'
        plt.savefig(filename)
        with open(filename, 'rb') as file:
            image = discord.File(file)
            await ctx.send(file=image)
        plt.close()
    else:
        print('Not enough data to show')
    
    
def filter_rows_by_keyword(df, keyword):
    df['Cleaned Title'] = df['Title'].str.lower().apply(lambda x: re.sub(r'[^\w\s]', '', x))
    cols = df.columns.tolist()
    cols.remove('Cleaned Title')
    cols.insert(1, 'Cleaned Title')
    df = df[cols]
    mask = df['Cleaned Title'].str.contains(keyword, case=False)
    filtered_df = df[mask]
    return filtered_df

def filter_dataframe_by_category(df, category_col, category):
    filtered_df = df[df[category_col] == category].copy()
    return filtered_df

async def plot_price_by_size(ctx, df, size_col, price_col):
    grouped_df = df.groupby(size_col).agg({price_col: 'mean', size_col: 'count'})
    grouped_df.rename(columns={size_col: 'Volume'}, inplace=True)
    sorted_df = grouped_df.sort_values(price_col)
    cmap = plt.cm.get_cmap('viridis')
    normalized_volume = (sorted_df['Volume'] - sorted_df['Volume'].min()) / (sorted_df['Volume'].max() - sorted_df['Volume'].min())
    colors = cmap(normalized_volume)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(sorted_df.index, sorted_df[price_col], color=colors)
    plt.xlabel('Size Category')
    plt.ylabel('Price')
    plt.title('Average Price by Size Category')
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(sorted_df['Volume'])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Volume')
    plt.xticks(rotation=45)
    plt.tight_layout()
    filename = 'sizes.png'
    plt.savefig(filename)
    with open(filename, 'rb') as file:
        image = discord.File(file)
        await ctx.send(file=image)
    plt.close()

def generate_volume(df):
    volume_df = df.groupby(['Year', 'Month']).size().reset_index(name='Volume')
    df = pd.merge(df, volume_df, on=['Year', 'Month'], how='left')
    return df

def predict_future_prices(df1, target_year, target_month):
    df = df1
    df = df.groupby([df['Relative Date'].dt.year.rename('Year'), df['Relative Date'].dt.month.rename('Month')])['Original Price'].mean().reset_index()
    df = generate_volume(df)  # Generate synthetic volume based on count of prices
    
    train_size = int(len(df) * 0.8)
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]
    
    scaler_price = MinMaxScaler()
    train_data_scaled_price = scaler_price.fit_transform(train_data[['Original Price']])
    test_data_scaled_price = scaler_price.transform(test_data[['Original Price']])

    scaler_volume = MinMaxScaler()
    train_data_scaled_volume = scaler_volume.fit_transform(train_data[['Volume']])
    test_data_scaled_volume = scaler_volume.transform(test_data[['Volume']])

    train_tensor = torch.from_numpy(np.column_stack((train_data_scaled_price, train_data_scaled_volume))).float()
    test_tensor = torch.from_numpy(np.column_stack((test_data_scaled_price, test_data_scaled_volume))).float()
    
    class LSTM(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(LSTM, self).__init__()
            self.hidden_size = hidden_size
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out
    
    input_size = 2  # Two input features: Original Price and Volume
    hidden_size = 32
    output_size = 1
    num_epochs = 100
    learning_rate = 0.001
    
    model2 = LSTM(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model2.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model2.train()
        optimizer.zero_grad()
        outputs = model2(train_tensor.unsqueeze(dim=0))
        loss = criterion(outputs.squeeze(), train_tensor[1:].squeeze())
        loss.backward()
        optimizer.step()
    
    model2.eval()
    with torch.no_grad():
        future_tensor = torch.zeros(1, 2)  # Initialize a tensor for future prediction
        future_tensor[0, 0] = test_data_scaled_price[-1, 0]  # Last observed price
        future_tensor[0, 1] = test_data_scaled_volume[-1, 0]  # Last observed volume
        
        predicted_prices = []  # List to store predicted prices for each future month
        for _ in range(target_month):
            future_prediction = model2(future_tensor.unsqueeze(dim=0))  # Make prediction for next month
            future_tensor = torch.cat((future_tensor[:, :1], future_prediction), dim=1)
            predicted_price = scaler_price.inverse_transform(future_prediction[:, 0].reshape(-1, 1)).squeeze()
            predicted_prices.append(predicted_price)
    
    best_prediction = np.mean(predicted_prices)
    locale.setlocale(locale.LC_ALL, '')  # Use the default system locale for formatting
    formatted_prediction = locale.currency(best_prediction, grouping=True)
    return formatted_prediction

titles = []
prices = []
old_prices = []
dates = []
old_dates = []
listing_images = []
listing_links = []
sizes = []

data = {
    'Title': titles,
    'Current Price': prices,
    'Original Price': old_prices,
    'Size': sizes,
    'Current Date': dates,
    'Original Date': old_dates,
    'Image Link': listing_images,
    'Listing Link': listing_links
}

categories = {
        'Tops': ['shirt', 'blouse', 't-shirt', 'tee', 'long-sleeve', 'longsleeve', 'long sleeve', 'short sleeve', 'sweater', 'tank', 'tank top', 'top', 'button-up', 'button-down', 'vest', 'polo', 'crop-top', 'box logo', 'sweatshirt'],
        'Bottoms': ['pants', 'jeans', 'flare', 'baggy', 'pant', 'cargo', 'talon-zip', 'dickies', 'painted denim', 'sweatpants', 'shorts', 'pleats please', 'joggers'],
        'Skirts': ['maxi', 'skirt', 'mini-skirt', 'pleated skirt', 'mini skirt', 'midi', 'midi skirt'],
        'Dresses': ['dress', 'gown'],
        'Shoes': ['loafer', 'shoes', 'sneakers', 'boots', 'jordan', 'air force one', 'chuck 70', 'guidi', 'rick owens ramones', 'dunk', 'gucci slides'],
        'Outerwear': ['Fur leather', 'Half zip', 'Quarter zip', 'Suit', 'Outerwear', 'Jacket', 'Puffer', 'jacket', 'coat', 'blazer', 'bomber', 'trenchcoat', 'trucker jacket', 'hoodie', 'zip-up', 'pullover', 'windbreaker', 'cardigan', 'Denim Trucker Jacket'],
        'Accessories': ['Sunglasses', 'Apron', 'Necklace', 'Watch', 'Socks', 'Tie', 'Bow tie', 'Purse', 'Ring', 'Gloves', 'belt', 
                      'Scarf', 'Umbrella', 'Boots', 'Mittens', 'Stockings', 'Earmuffs', 'Hair band', 'Safety pin', 'Watch', 'Hat', 'Beanie', 'Cap', 'Beret', 'card holder', 'Straw hat', 'Derby hat', 'Helmet', 'Top hat', 'Mortar board']
}
model1 = load_model('model.h5')
bot.run("TOKEN")


