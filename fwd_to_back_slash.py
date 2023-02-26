import csv
import os
import tkinter as tk
from tkinter import filedialog

# Define a function to convert slashes based on user input
def convert_slashes(cell_value, replace_fwd):
    if replace_fwd:
        return cell_value.replace("/", "\\")
    else:
        return cell_value.replace("\\", "/")

# Determine mode of operation
mode = input("Select mode of operation:\n1. Headless mode\n2. Desktop mode\n")

if mode == "1":
    # Get the path to the CSV file from the user
    path = input("Enter the path to the CSV file: ")
else:
    # Open a file dialog to select the CSV file
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    path = os.path.abspath(file_path)

# Ask the user if they want to convert "/" to "\" or vice-versa
replace_fwd = input("Do you want to convert '/' to '\\' (enter 'y' for yes or 'n' for no)? ")

# Open the CSV file and create a new file for the edited data
if replace_fwd.lower() == 'y':
    output_filename = os.path.splitext(path)[0] + '_bck.csv'
else:
    output_filename = os.path.splitext(path)[0] + '_fwd.csv'

with open(path, "r") as input_file, open(output_filename, "w", newline="") as output_file:
    reader = csv.reader(input_file)
    writer = csv.writer(output_file)

    # Iterate over each row in the CSV file
    for i, row in enumerate(reader):
        # Iterate over each cell in the row
        for j, cell in enumerate(row):
            # Replace slashes in the cell based on user input, if it's in the first column
            if j == 0:
                row[j] = convert_slashes(cell, replace_fwd.lower() == 'y')

        # Write the edited row to the new file
        writer.writerow(row)

        # Print the first three lines of the CSV file to the terminal after editing the path column
        if i < 3:
            print(row)

if replace_fwd.lower() == 'y':
    print("CSV file path column edited successfully, '/' converted to '\\'!")
else:
    print("CSV file path column edited successfully, '\\' converted to '/'!")
print(f'Saved output as {output_filename} in input folder.')
