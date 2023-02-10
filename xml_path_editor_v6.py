"""
This script was written by: ChatGPT
Prompted by: Marc Ivan Manalac
    
Description:
This script is a tool that allows a user to edit XML files in bulk. The user has the option to run the script in either headless mode or desktop mode.
In headless mode, the user is prompted to enter the path of the XML folder and the JPG folder, and the script will perform the edits.
In desktop mode, the user is prompted to browse and select the directory of the XML files and the JPG files using file dialog boxes in a GUI environment.
The script performs the following steps for each XML file in the selected directory:
- Creates a file path by joining the root directory and the file name.
- Parses the XML file.
- Retrieves the root of the XML file.
- Gets the file name without the extension.
- Creates a jpg path.
- Finds the path element in the XML file and updates it with the jpg_path.
- Writes the changes back to the XML file.
- Prints the current count and the total number of files processed.
- Finally, the script outputs a message indicating that the edit is complete.
"""
import os
import tkinter as tk
from tkinter import filedialog
from xml.etree import ElementTree as ET

def browse_directory():
    root = tk.Tk()
    root.withdraw()
    xml_directory = filedialog.askdirectory(parent=root, title='Choose the directory of the XML files')
    jpg_directory = filedialog.askdirectory(parent=root, title='Choose the directory of the JPG files')
    batch_edit_xml(xml_directory, jpg_directory)

def headless_mode():
    xml_directory = input("Enter the path of the XML folder: ")
    jpg_directory = input("Enter the path of the JPG folder: ")
    batch_edit_xml(xml_directory, jpg_directory)

def batch_edit_xml(xml_directory, jpg_directory):
    count = 1 # initializing count to 1
    for root, dirs, files in os.walk(xml_directory):
        for file in files:
            if file.endswith(".xml"):
                file_path = os.path.join(root, file) # creating a file path by joining the root and the file name
                xml_tree = ET.parse(file_path) # parsing the XML file
                xml_root = xml_tree.getroot() # getting the root of the XML file
                filename = os.path.splitext(file)[0] # getting the file name without the extension
                jpg_path = os.path.join(jpg_directory, os.path.basename(root), filename + '.jpg') # creating a jpg path
                xml_root.find('./path').text = jpg_path # finding the path element in the XML file and updating it with the jpg_path
                xml_tree.write(file_path) # writing the changes back to the XML file
                print(f"{count} of {len(files)}: {file_path}") # printing the current count and the total number of files processed
                count += 1
                if count > len(files): # checking if the count has reached the length of the files
                    count = 1 # resetting the count back to 1
    print("Edit Complete") # indicating that the edit is complete

mode = input("Enter 1 for headless mode or 2 for desktop mode: ")
if mode == '1':
    headless_mode()
elif mode == '2':
    browse_directory()
else:
    print("Invalid input. Please enter 1 or 2.")
