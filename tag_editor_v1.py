import os
import tkinter as tk
from tkinter import filedialog
from xml.etree import ElementTree as ET

def browse_directory():
    root = tk.Tk()
    root.withdraw()
    xml_directory = filedialog.askdirectory(parent=root, title='Choose the directory of the XML files')
    search_tag = input("Enter the name of the tag to search for: ")
    replace_value = input("Enter the new value to replace with: ")
    batch_edit_xml(xml_directory, search_tag, replace_value)

def headless_mode():
    xml_directory = input("Enter the path of the XML folder: ")
    search_tag = input("Enter the name of the tag to search for: ")
    replace_value = input("Enter the new value to replace with: ")
    batch_edit_xml(xml_directory, search_tag, replace_value)

def batch_edit_xml(xml_directory, search_tag, replace_value):
    count = 1 # initializing count to 1
    for root, dirs, files in os.walk(xml_directory):
        for file in files:
            if file.endswith(".xml"):
                file_path = os.path.join(root, file) # creating a file path by joining the root and the file name
                xml_tree = ET.parse(file_path) # parsing the XML file
                xml_root = xml_tree.getroot() # getting the root of the XML file
                for tag in xml_root.findall('.//' + search_tag):
                    tag.text = replace_value
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
