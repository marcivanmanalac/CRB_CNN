import argparse
import csv
import os
import tkinter as tk
from tkinter import filedialog
import xml.etree.ElementTree as ET


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=int, required=True, help="1 for headless mode, 2 for desktop mode")
    return parser.parse_args()


def get_dirs(args):
    if args.mode == 1:
        xml_dir = input("Enter the path to the xml directory: ")
        csv_dir = input("Enter the path to the csv output directory: ")
    elif args.mode == 2:
        root = tk.Tk()
        root.withdraw()
        xml_dir = filedialog.askdirectory(title="Select the xml directory")
        csv_dir = filedialog.askdirectory(title="Select the csv output directory")
    else:
        raise ValueError("Invalid mode. Please enter 1 for headless mode, or 2 for desktop mode.")

    class_names = input("Enter class names separated by commas: ").split(",")
    labels = input("Enter labels separated by commas: ").split(",")
    return xml_dir, csv_dir, class_names, labels


def process_xml_file(xml_file, class_names, labels):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    data = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name in class_names:
            label = labels[class_names.index(name)]
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            path = root.find("path").text  # get the path tag
            #trying to fix CSV output
            #data.append((path, xmin, ymin, xmax, ymax, label))
            data.append((path, xmin, ymin, xmax, ymax, label))

    return data


def process_xml_files(xml_dir, csv_dir, class_names, labels):
    csv_file = os.path.join(csv_dir, "annotations.csv")
    with open(csv_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "path", "xmin", "ymin", "xmax", "ymax", "label"])  # add "path" column
        for root, dirs, files in os.walk(xml_dir):
            for file in files:
                if file.endswith(".xml"):
                    xml_file = os.path.join(root, file)
                    data = process_xml_file(xml_file, class_names, labels)
                    if data:
                        filename = os.path.splitext(file)[0] + ".jpg"
                        for row in data:
                            #EDIT
                            #writer.writerow(list(row))  # include path value as first element
                            writer.writerow([filename] + list(row))
def main():
    args = parse_args()
    xml_dir, csv_dir, class_names, labels = get_dirs(args)
    process_xml_files(xml_dir, csv_dir, class_names, labels)
    print("CSV file saved to: ", os.path.join(csv_dir, "annotations.csv"))


if __name__ == "__main__":
    main()
