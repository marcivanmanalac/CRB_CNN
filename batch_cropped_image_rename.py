# This Python script batch edits a directory with multiple subdirectories containing JPG files. 
# The script renames all JPG files within each subdirectory so that the JPG file name is prefixed with 
# the immediate folder name followed by an underscore and ending with the original JPG name. The script 
# supports both headless mode and desktop mode using Tkinter for accepting user input. In addition, the script 
# uses tqdm to show the progress of the batch editing process with a progress bar and displays print statements to 
# provide feedback to the user throughout the process.

# to run headless type "python script.py --headless"
import os
import argparse
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm


def batch_edit_directory(directory):
    print(f"Batch editing directory {directory}...\n")
    # Loop through all subdirectories
    for root, dirs, files in os.walk(directory):
        for file in tqdm(files, desc=f"Processing directory {root}"):
            # Only process JPG files
            if file.lower().endswith('.jpg'):
                # Get the immediate folder name
                folder_name = os.path.basename(root)
                # Rename the JPG file
                old_path = os.path.join(root, file)
                new_file_name = f"{folder_name}_{file}"
                new_path = os.path.join(root, new_file_name)
                os.rename(old_path, new_path)


def get_input_directory():
    root = tk.Tk()
    root.withdraw()
    input_directory = filedialog.askdirectory(title='Select the input cropped image directory to edit')
    return input_directory


def get_args():
    parser = argparse.ArgumentParser(description='Batch edit directory with multiple subdirectories')
    parser.add_argument('--headless', action='store_true', help='Headless mode without GUI')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # Get input directory from GUI or terminal
    if args.headless:
        input_directory = input("Enter the input directory: ")
    else:
        input_directory = get_input_directory()

    # Batch edit the directory
    batch_edit_directory(input_directory)

    print("\nBatch edit complete.")
