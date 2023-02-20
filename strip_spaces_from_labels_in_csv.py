'''
This Python script fixes the "label" header's column values of a user-inputted CSV file by removing any spaces after the comma-separated values. 
The script asks the user whether they want to run the script in headless or desktop mode. In headless mode, the user is prompted to enter the filename of the CSV file to fix. 
In desktop mode, a file dialog is opened using tkinter to allow the user to select the CSV file to fix. Once the user provides the filename, the script reads the CSV file using the csv module, 
fixes the "label" values of each row, and overwrites the original file with the updated values. 
A message is then printed to the console indicating that the CSV file has been updated.
'''
import csv
import tkinter as tk
from tkinter import filedialog


def fix_label_values(filename):
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            for key in row:
                row[key] = row[key].strip()
            rows.append(row)

    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    # Ask user to choose mode
    while True:
        mode = input("Enter 1 for headless mode or 2 for desktop mode: ")
        if mode == '1':
            # Get filename from user
            filename = input("Enter CSV filename: ")
            fix_label_values(filename)
            print("CSV file has been updated.")
            break
        elif mode == '2':
            # Create tkinter window to open file dialog
            root = tk.Tk()
            root.withdraw()

            # Open file dialog to choose CSV file
            filename = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])

            if filename:
                fix_label_values(filename)
                print("CSV file has been updated.")
                break
            else:
                print("No file selected.")
        else:
            print("Invalid mode entered. Please try again.")


if __name__ == "__main__":
    main()
