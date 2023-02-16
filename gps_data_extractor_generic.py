import os
import csv
from PIL import Image
import piexif

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

            # get the GPS data from the image metadata using piexif
            exif_data = piexif.load(image.info['exif'])
            gps_data = exif_data['GPS']

            # convert the GPS data to decimal degrees
            latitude = float(gps_data[2][0][0]) / float(gps_data[2][0][1])
            longitude = float(gps_data[4][0][0]) / float(gps_data[4][0][1])
            altitude = float(gps_data[6][0]) / float(gps_data[6][1])

            # get the class label from the file path
            class_label = os.path.basename(root)

            # update the class count
            if class_label in class_counts:
                class_counts[class_label] += 1
            else:
                class_counts[class_label] = 1

            # add the image path, class label, and GPS data to the data list
            data.append([file_path, class_label, latitude, longitude, altitude])

# create the CSV file
with open('data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['path', 'class', 'latitude', 'longitude', 'altitude'])
    for row in data:
        # write the image path, class label, and GPS data to the CSV file
        writer.writerow(row)

# print the class counts
print(class_counts)
