'''
This script takes a CSV file containing a dataset, and splits the data into three separate CSV files for training, validation, and testing sets.
The proportions of the split are set to 60% for training, 30% for validation, and 10% for testing. 
The user is prompted via a Tkinter window to select the input CSV file and the output directory for the split files. 
The output files are saved in the selected output directory with the names 'train.csv', 'val.csv', and 'test.csv'. 
The script shuffles the data randomly before splitting it to ensure that the splits are representative of the whole dataset. 
This script was created by me, ChatGPT, a large language model trained by OpenAI.
Prompted by Marc Ivan Manalac

- Also allows headless mode, now.
'''

import csv
import random
import os

# Check if running in headless mode
if os.environ.get('DISPLAY', '') == '':
    # Running headless
    input_file_path = input("Enter the path to the CSV file: ")
    output_dir_path = input("Enter the path to the output directory: ")
else:
    # Running in desktop mode
    import tkinter as tk
    from tkinter import filedialog
    
    # Set up the Tkinter window for selecting the input and output files
    root = tk.Tk()
    root.withdraw()

    # Ask the user to select the input CSV file
    input_file_path = filedialog.askopenfilename(title="Select input CSV file",
                                                 filetypes=[("CSV files", "*.csv")])
    if not input_file_path:
        print("No input file selected.")
        exit()

    # Ask the user to select the output directory for the split files
    output_dir_path = filedialog.askdirectory(title="Select output directory")
    if not output_dir_path:
        print("No output directory selected.")
        exit()

# Path to the output CSV files for the training, validation, and testing sets
train_csv_file = os.path.join(output_dir_path, 'train.csv')
val_csv_file = os.path.join(output_dir_path, 'val.csv')
test_csv_file = os.path.join(output_dir_path, 'test.csv')

# Proportions of the split (train: 60%, val: 30%, test: 10%)
train_split = 0.6
val_split = 0.3
test_split = 0.1

# Read the CSV file and split the rows randomly
with open(input_file_path, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)  # Skip the header row

    # Shuffle the rows randomly
    rows = list(reader)
    random.shuffle(rows)

    # Compute the number of rows for each split
    num_rows = len(rows)
    num_train = int(num_rows * train_split)
    num_val = int(num_rows * val_split)
    num_test = num_rows - num_train - num_val

    # Write the rows to the output CSV files
    with open(train_csv_file, 'w', newline='') as train_f, \
            open(val_csv_file, 'w', newline='') as val_f, \
            open(test_csv_file, 'w', newline='') as test_f:
        train_writer = csv.writer(train_f)
        val_writer = csv.writer(val_f)
        test_writer = csv.writer(test_f)

        # Write the header row to all output CSV files
        train_writer.writerow(header)
        val_writer.writerow(header)
        test_writer.writerow(header)

        # Write the rows to the output CSV files
        for i, row in enumerate(rows):
            if i < num_train:
                train_writer.writerow(row)
            elif i < num_train + num_val:
                val_writer.writerow(row)
            else:
                test_writer.writerow(row)
                
# Remove blank rows from the output CSV files
for file_path in [train_csv_file, val_csv_file, test_csv_file]:
    with open(file_path, 'r') as f:
        rows = list(csv.reader(f))
    
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            if any(row):
                writer.writerow(row)
                
print("Finished!")
