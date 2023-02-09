###
# This was written using a ChatGPT Prompt
# Description: Primarily for preprocessing image data for CNN. This script is for Editing XML path tags to have their corresponding JPG paths. Ensure JPG directories are named the same as XML counterpart and within a directory ending in '_Cropped'
# By: Marc Ivan Manalac
# ###

import os
import xml.etree.ElementTree as ET
from tkinter import Tk
from tkinter import filedialog

# function to edit the XML files
def edit_xml(file_path, jpg_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    path_tag = root.find("path")
    path_tag.text = jpg_path
    tree.write(file_path)

# function to batch edit all XML files in a directory
def batch_edit_xml(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".xml"):
                file_path = os.path.join(root, file)
                jpg_file = file.replace(".xml", ".jpg")
                jpg_path = os.path.join(root, jpg_file)
                if os.path.exists(jpg_path):
                    count += 1
                    print("Editing XML:", file_path)
                    print("JPG path:", jpg_path)
                    print(f"{count} of ???")
                    edit_xml(file_path, jpg_path)
    return count

# function to allow the user to select a directory
def select_directory():
    root = Tk()
    root.withdraw()
    return filedialog.askdirectory()

# main function to run the program
if __name__ == "__main__":
    directory = select_directory()
    if directory:
        count = batch_edit_xml(directory)
        print(f"Edit Complete, {count} XML files edited.")
