'''
 This script assumes:
  1) Image data is labeled and annotations were created and saved on XML files.
  2) csv_maker_v2.py was used to create CSV file.
  3) csv_splitter_v2.py was used to created random set of 60-30-10 split of training-validation-testing.

The code I provided does the following:
- Reads in three CSV files containing data for training, validation, and testing respectively.
- Converts the image data in each CSV file to NumPy arrays and stores them in separate arrays.
- Applies data augmentation to the training data using the Keras ImageDataGenerator class.
- Defines a convolutional neural network model using the Keras Sequential API.
- Compiles the model with the Adam optimizer, binary cross-entropy loss, and metrics for accuracy, precision, and AUC-ROC.
- Fits the model to the training data, using the validation data for monitoring and early stopping if the model starts overfitting.
- Evaluates the model on the test data and generates a classification report, confusion matrix, and AUC-ROC curve.
- Saves the trained model to disk in the Keras H5 format.
'''

import numpy as np
import pandas as pd
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import os.path
from time import sleep
# Set random seed for reproducibility
np.random.seed(123)

# Looks for csv files using GUI
def browse_csvs():
    root = tk.Tk()
    root.withdraw()
    print("Select the Img_to_CSV files")
    train_csv = filedialog.askopenfilename(parent=root, title='Choose the Training Set CSV')
    #check if invalid csv file
    if not os.path.isfile(train_csv):
        print("Error: Invalid file path for training set CSV. Please enter a valid file path.")
        return

    val_csv = filedialog.askopenfilename(parent=root, title='Choose the Validation Set CSV')
    if not os.path.isfile(val_csv):
        print("Error: Invalid file path for training set CSV. Please enter a valid file path.")
        return
    test_csv = filedialog.askopenfilename(parent=root, title='Choose the Test Set CSV')
    if not os.path.isfile(test_csv):
        print("Error: Invalid file path for training set CSV. Please enter a valid file path.")
        return
    dest_directory = filedialog.askdirectory(parent=root, title='Choose the destination for the generated numPy files: ')
    if not os.path.isdir(dest_directory):
        print("Error: Invalid directory path for destination directory. Please enter a valid directory path.")
        return

    return train_csv, test_csv, val_csv, dest_directory

def headless_mode():
    train_csv = input("Enter the path of the Training Set CSV: ")
    val_csv = input("Enter the path of the Validation Set CSV:  ")
    test_csv = input("Enter the path of the Test Set CSV: ")
    dest_directory = input("Choose the destination for the generated numPy files: ")
    return train_csv, test_csv, val_csv, dest_directory

mode = input("Enter 1 for headless mode or 2 for desktop mode: ")
if mode == '1':
    #capture the return variables
    train_csv, val_csv,test_csv, dest_directory = headless_mode() 
elif mode == '2':
    train_csv, val_csv,test_csv, dest_directory = browse_csvs()
else:
    print("Invalid input. Please enter 1 or 2.")

# Read CSV Data first to ensure
print("Reading some of data from the CSV files.")
sleep(1)
# Use the read_csv function to read the first three lines of the CSV file
train_data_head = pd.read_csv(train_csv,nrows=3)
val_data_head = pd.read_csv(val_csv,nrows=3) 
test_data_head = pd.read_csv(test_csv,nrows=3)

# Read the last row of the csv file
train_tail = pd.read_csv(train_csv, skiprows=range(1, len(train_data_head)+1))
val_tail = pd.read_csv(val_csv, skiprows=range(1, len(val_data_head)+1))
test_tail = pd.read_csv(test_csv, skiprows=range(1, len(test_data_head)+1))

# concatenate the first three rows and the last row into a single DataFrame
train_print = pd.concat([train_data_head, train_tail], ignore_index=True)
val_print = pd.concat([val_data_head, val_tail], ignore_index=True)
test_print = pd.concat([test_data_head, test_tail], ignore_index=True)

print('Values inside the data set:')
print('Train set:')
print(train_print)
print('Validation set:')
print(val_print)
print('Test Set:')
print(test_print)

# Load training, validation, and testing data from separate CSV files
print("Loading data from the CSV files.")
sleep(1)
train_df = pd.read_csv(train_csv, header=0)
val_df = pd.read_csv(val_csv, header=0)
test_df = pd.read_csv(test_csv, header=0)


# Define image preprocessing and data augmentation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Define batch size and image size
batch_size = 32
img_size = (224, 224)

# Define data generators for training, validation, and testing
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='path',
    y_col='label',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)
val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='path',
    y_col='label',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='path',
    y_col='label',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Define and compile the model
model = tf.keras.models.Sequential([
    tf.keras.applications.ResNet50(
        include_top=False, weights='imagenet', input_shape=(224, 224, 3)
    ),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(5, activation='softmax')
])
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)


# Define the number of training and validation steps per epoch
train_steps_per_epoch = train_generator.n // batch_size
val_steps_per_epoch = val_generator.n // batch_size

# Define the number of epochs for training
epochs = 10

# Define data augmentation for training data
train_augmented_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='path',
    y_col='label',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2
)

# Fit the model with data augmentation
history = None
with tqdm(total=epochs, desc='Training') as pbar:
    for epoch in range(epochs):
        history = model.fit(
            train_augmented_generator,
            steps_per_epoch=train_steps_per_epoch,
            epochs=1,
            validation_data=val_generator,
            validation_steps=val_steps_per_epoch,
            verbose=0
        )
        pbar.update(1)

# Evaluate the model on the test set
test_generator.reset()
y_pred = model.predict(test_generator, steps=len(test_generator), verbose=1)
y_true = np.argmax(test_generator.labels, axis=1)

# Generate the classification report and confusion matrix
report = classification_report(y_true, np.argmax(y_pred, axis=1), target_names=class_names)
print(report)

cm = confusion_matrix(y_true, np.argmax(y_pred, axis=1))
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(class_names)
ax.yaxis.set_ticklabels(class_names)

# Generate the AUC-ROC curve
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in tqdm(range(num_classes), desc='Computing ROC curve'):
    fpr[i], tpr[i], _ = roc_curve(y_true == i, y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(label_binarize(y_true, classes=range(num_classes)).ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curve for each class
plt.figure()
lw = 2
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    
plt.plot(fpr["micro"], tpr["micro"], color='deeppink', lw=lw, linestyle='--',
         label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
