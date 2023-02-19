'''
This Python script prompts the user to select the mode of operation: headless mode or desktop mode. 
In headless mode, the user is prompted to enter the path to a CSV file. 
In desktop mode, the user can select the CSV file through a file dialog.

Once the CSV file has been selected, the script opens the file, reads its content, 
and replaces all forward slashes in the first column of each row with backslashes. 
The edited rows are then written to a new CSV file called "edited_data.csv".

After editing the first column of each row, the script prints the first three rows 
of the edited CSV file to the terminal. 
Finally, the script outputs a message indicating that the CSV file path column 
has been successfully edited.
'''
import csv
import os
import tkinter as tk
from tkinter import filedialog

# Define a function to convert slashes based on user input
def convert_slashes(path, replace_fwd):
    if replace_fwd:
        return path.replace("/", "\\")
    else:
        return path.replace("\\", "/")

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
    output_filename = os.path.splitext(path)[0] + '_fwd.csv'
else:
    output_filename = os.path.splitext(path)[0] + '_bck.csv'

with open(path, "r") as input_file, open(output_filename, "w", newline="") as output_file:
    reader = csv.reader(input_file)
    writer = csv.writer(output_file)

    # Read and discard the column names row
    column_names = next(reader)
    writer.writerow(column_names)
    print(column_names)

    # Iterate over each row in the CSV file
    for i, row in enumerate(reader):
        # Replace slashes in the path column based on user input
        row[1] = convert_slashes(row[1], replace_fwd.lower() == 'y')

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