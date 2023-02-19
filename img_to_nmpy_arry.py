import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import preprocessing
import tkinter as tk
from tkinter import filedialog
from time import sleep
from tqdm import tqdm
import os.path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0

# Looks for csv files using GUI
def browse_csvs():
    root = tk.Tk()
    root.withdraw()
    print("Select the Img_to_CSV files")
    train_csv = filedialog.askopenfilename(parent=root, title='Choose the Training Set CSV')
    #check if invalid csv file
    if not os.path.isfile(train_csv):
        print("Error: Invalid file path for training set CSV. Please enter a valid file path.")
        return

    val_csv = filedialog.askopenfilename(parent=root, title='Choose the Validation Set CSV')
    if not os.path.isfile(val_csv):
        print("Error: Invalid file path for training set CSV. Please enter a valid file path.")
        return
    test_csv = filedialog.askopenfilename(parent=root, title='Choose the Test Set CSV')
    if not os.path.isfile(test_csv):
        print("Error: Invalid file path for training set CSV. Please enter a valid file path.")
        return
    dest_directory = filedialog.askdirectory(parent=root, title='Choose the destination for the generated numPy files: ')
    if not os.path.isdir(dest_directory):
        print("Error: Invalid directory path for destination directory. Please enter a valid directory path.")
        return

    return train_csv, test_csv, val_csv, dest_directory

def headless_mode():
    train_csv = input("Enter the path of the Training Set CSV: ")
    val_csv = input("Enter the path of the Validation Set CSV:  ")
    test_csv = input("Enter the path of the Test Set CSV: ")
    dest_directory = input("Choose the destination for the generated numPy files: ")
    return train_csv, test_csv, val_csv, dest_directory

mode = input("Enter 1 for headless mode or 2 for desktop mode: ")
if mode == '1':
    #capture the return variables
    train_csv, val_csv,test_csv, dest_directory = headless_mode() 
elif mode == '2':
    train_csv, val_csv,test_csv, dest_directory = browse_csvs()
else:
    print("Invalid input. Please enter 1 or 2.")


# Read CSV Data first to ensure
print("Reading some of data from the CSV files.")
sleep(1)
# Use the read_csv function to read the first three lines of the CSV file
train_data_head = pd.read_csv(train_csv,nrows=3)
val_data_head = pd.read_csv(val_csv,nrows=3) 
test_data_head = pd.read_csv(test_csv,nrows=3)

# Read the last row of the csv file
train_tail = pd.read_csv(train_csv, skiprows=range(1, len(train_data_head)+1))
val_tail = pd.read_csv(val_csv, skiprows=range(1, len(val_data_head)+1))
test_tail = pd.read_csv(test_csv, skiprows=range(1, len(test_data_head)+1))

# concatenate the first three rows and the last row into a single DataFrame
train_print = pd.concat([train_data_head, train_tail], ignore_index=True)
val_print = pd.concat([val_data_head, val_tail], ignore_index=True)
test_print = pd.concat([test_data_head, test_tail], ignore_index=True)

print('Values inside the data set:')
print('Train set:')
print(train_print)
print('Validation set:')
print(val_print)
print('Test Set:')
print(test_print)
 
# Load data from CSV files
print("Loading data from the CSV files.")
sleep(1)
train_data = pd.read_csv(train_csv, dtype={'filename': str, 'path': str, 'xmin': int, 'ymin': int, 'xmax': int, 'ymax': int, 'label': str}, na_values=['NA', 'NaN'])
val_data = pd.read_csv(val_csv, dtype={'filename': str, 'path': str, 'xmin': int, 'ymin': int, 'xmax': int, 'ymax': int, 'label': str}, na_values=['NA', 'NaN'])
test_data = pd.read_csv(test_csv, dtype={'filename': str, 'path': str, 'xmin': int, 'ymin': int, 'xmax': int, 'ymax': int, 'label': str}, na_values=['NA', 'NaN'])
print("Data loading complete!")

# Create labels using labels column in CSV
train_labels = np.array(train_data['label'])
val_labels = np.array(val_data['label'])
test_labels = np.array(test_data['label'])

# Create NumPy arrays of images for each set
print("Creating Numpy array of images for each set" )
sleep(1)
try:
    train_images = []
    for i, row in tqdm(train_data.iterrows(), total=len(train_data)):
        path = row['path']
        # Load image and turn into array
        img_pil=tf.keras.preprocessing.image.load_img(path)
        img = tf.keras.preprocessing.image.array_to_img(img_pil)
        train_images.append(img)
    train_images = np.array(train_images)

    val_images = []
    for i, row in tqdm(val_data.iterrows(), total=len(val_data)):
        path = row['path']
        img_pil=tf.keras.preprocessing.image.load_img(path)
        img = tf.keras.preprocessing.image.array_to_img(img_pil)
        val_images.append(img)
    val_images = np.array(val_images)

    test_images = []
    for i, row in tqdm(test_data.iterrows(), total=len(test_data)):
        path = row['path']
        img_pil=tf.keras.preprocessing.image.load_img(path)
        img = tf.keras.preprocessing.image.array_to_img(img_pil)
        test_images.append(img)
    test_images = np.array(test_images)

    #Print shape of NumPy arrays
    print("Shape of train_images:", train_images.shape)
    print("Shape of train_labels:", train_labels.shape)
    print("Shape of val_images:", val_images.shape)
    print("Shape of val_labels:", val_labels.shape)
    print("Shape of test_images:", test_images.shape)
    print("Shape of test_labels:", test_labels.shape)

    # Print the first three elements of each NumPy array
    print("First three elements of train_images: ")
    print(train_images[:3])

    print("First three elements of val_images: ")
    print(val_images[:3])

    print("First three elements of test_images: ")
    print(test_images[:3])


    # Save Numpy Arrays
    print(f"Saving NumPy Arrays for later use in: {dest_directory}")
    sleep(1)
    np.save('train_images.npy', train_images)
    np.save('val_images.npy', val_images)
    np.save('test_images.npy', test_images)

    print(f"Complete. Saved in {dest_directory}")
    
except Exception as e:
    print("Error:", e)
    print("An error occurred while creating or saving the NumPy arrays. Please check that the input files and destination directory are correct.")
