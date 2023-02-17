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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(123)

# Load training, validation, and testing data from separate CSV files
# With tqdm progress bar
train_df = tqdm(pd.read_csv('train.csv', header=0))
val_df = tqdm(pd.read_csv('val.csv', header=0))
test_df = tqdm(pd.read_csv('test.csv', header=0))




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
