# The code is a Python script that takes two command line arguments - 
# the path to a CSV file and the path to a directory containing images. 
# It reads the CSV file into a pandas dataframe and iterates over the first column of the dataframe, 
# which contains names of jpg images. For each row, it constructs the absolute path to the corresponding image in the input directory and replaces 
# the value in the first column of the dataframe with this absolute path. Finally, it saves the modified dataframe to a new CSV file.

# Overall, the code is used to update a CSV file with the absolute paths to jpg images in a given directory, 
# based on their filename (which is assumed to match the value in the first column of the CSV file).

# example execution in colab
# !python csv_first_col_path_editor.py --csv_file '/content/drive/MyDrive/input.csv' --image_dir '/content/drive/MyDrive/images/'
import os
import pandas as pd
import argparse

# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--csv_file', type=str, help='Path to CSV file with image names')
parser.add_argument('--image_dir', type=str, help='Path to directory with images')
parser.add_argument('--output_dir', type=str, help='Path to save directory')
args = parser.parse_args()

# Read CSV file into a pandas dataframe
df = pd.read_csv(args.csv_file,header=None) # specify header number if there are headers

# Iterate over the first column of the dataframe
for idx, row in df.iterrows():
    image_name = row[0]  # Get the name of the image from the first column
    image_path = os.path.join(args.image_dir, image_name)  # Construct the absolute path to the image
    df.at[idx, df.columns[0]] = image_path  # Replace the value in the dataframe with the absolute path

# Save the modified dataframe to a new CSV file in the specified output directory
output_path = os.path.join(args.output_dir, 'output.csv')
df.to_csv(output_path, index=False)

print (f'Edit Complete. Saved to {output_path}')