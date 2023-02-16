'''In this script, we first define the list of image extensions to search for, as well as the list of classes to search for. 
Then, we use the filedialog module from the tkinter library to allow the user to select a directory.
Next, we create a pandas dataframe to store the image paths, classes, and counts. 
We then recursively search for images in the directory and its subdirectories using the os.walk method. 
For each image, we check if it has one of the image extensions we're looking for, and if so, we extract the image class from
the filename by looking for the classes in the filename.
We then add the image path and class to the dataframe, and also update the counts for each class. 
Finally, we save the dataframe to a CSV file in the selected directory.
Note that this script assumes that the images are named in a way that allows you to extract the class from the filename, 
and that the directory structure follows the convention of having all images for each class in a separate directory. 
You may need to modify the script to handle different naming conventions or directory structures.'''
import os
import pandas as pd
from tkinter import filedialog, Tk

# Define the list of image extensions to search for
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.raw', '.heic']

# Define the list of classes to search for
CLASSES = ['oak', 'maple', 'birch', 'pine']

# Get the directory from the user
root = Tk()
root.withdraw()
folder_selected = filedialog.askdirectory()

# Create a pandas dataframe to store the image paths, classes, and counts
data = {'path': [], 'class': []}
for c in CLASSES:
    data[c] = []
df = pd.DataFrame(data)

# Recursively search for images in the directory and its subdirectories
for dirpath, dirnames, filenames in os.walk(folder_selected):
    for file in filenames:
        # Check if the file is an image
        if any(file.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
            # Get the image class from the filename
            for c in CLASSES:
                if c in file.lower():
                    # Add the image path and class to the dataframe
                    path = os.path.join(dirpath, file)
                    df = df.append({'path': path, 'class': c}, ignore_index=True)
                    df.loc[df['class'] == c, c] = len(df.loc[df['class'] == c])

# Save the dataframe to a CSV file
df.to_csv(os.path.join(folder_selected, 'image_data.csv'), index=False)
