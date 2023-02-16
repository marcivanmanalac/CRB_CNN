#requires dji SDK
import os
import csv
from PIL import Image
from dji_sdk import telemetry

# set up DJI SDK
api_key = 'your_api_key'
telemetry.init(api_key)

# specify the directory to search for images
dir_path = '/path/to/images'

# specify the file extensions to search for
extensions = ('.jpg', '.png', '.raw', '.heic')

# create a dictionary to store the class counts
class_counts = {}

# create a list to store the image paths and class labels
data = []

# loop through all files in the directory and its subdirectories
for root, dirs, files in os.walk(dir_path):
    for file in files:
        # check if the file has a valid extension
        if file.endswith(extensions):
            # get the file path and open the image
            file_path = os.path.join(root, file)
            image = Image.open(file_path)

            # get the GPS data from the DJI SDK
            gps_data = telemetry.get_gps_data(file_path)

            # get the class label from the file path
            class_label = os.path.basename(root)

            # update the class count
            if class_label in class_counts:
                class_counts[class_label] += 1
            else:
                class_counts[class_label] = 1

            # add the image path, class label, and GPS data to the data list
            data.append([file_path, class_label, gps_data])

# create the CSV file
with open('data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['path', 'class', 'latitude', 'longitude', 'altitude'])
    for row in data:
        # write the image path, class label, and GPS data to the CSV file
        writer.writerow([row[0], row[1], row[2]['latitude'], row[2]['longitude'], row[2]['altitude']])

# print the class counts
print(class_counts)
