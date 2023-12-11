# Import necessary libraries
import matplotlib
matplotlib.use('TKAgg')
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop

# Set the base directory and define directories for training and validation data
base_dir = '/data/train1'
train_dir = os.path.join(base_dir, 'data/')  # Training data directory
validation_dir = os.path.join(base_dir, 'validation/')  # Validation data directory
train_nsnows_dir = os.path.join(train_dir, 'no')  # Directory for training non-snow pictures
train_snows_dir = os.path.join(train_dir, 'snow')  # Directory for training snow pictures
validation_nsnows_dir = os.path.join(validation_dir, 'no')  # Directory for validation non-snow pictures
validation_snows_dir = os.path.join(validation_dir, 'snow')  # Directory for validation snow pictures

# Set batch size, number of epochs, and image dimensions
batch_size = 64
epochs = 80
IMG_HEIGHT = 128
IMG_WIDTH = 128

# Count the number of images in each training and validation category
num_nsnows_tr = len(os.listdir(train_nsnows_dir))
num_snows_tr = len(os.listdir(train_snows_dir))
num_nsnows_val = len(os.listdir(validation_nsnows_dir))
num_snows_val = len(os.listdir(validation_snows_dir))
total_train = num_nsnows_tr + num_snows_tr
total_val = num_nsnows_val + num_snows_val

# Function to plot images
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Create an image data generator for training with augmentation
image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=5,
    width_shift_range=.1,
    height_shift_range=.1,
    horizontal_flip=True,
    zoom_range=0.1
)

# Create a data generator for training images
train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='binary')

# Display augmented images for visualization
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

# Create a data generator for validation images
image_gen_val = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                 directory=validation_dir,
                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                 class_mode='binary')

# Create a convolutional neural network model
model_new = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu',
                           input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.3),  # 0.4
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.3),  # 0.3
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model with optimizer, loss function, and metrics
model_new.compile(optimizer=RMSprop(lr=0.005),
                  loss='binary_crossentropy',
                  metrics=['acc'])

# Display model summary
model_new.summary()

# Print trainable variables
print(model_new.trainable_variables)

# Train the model using the augmented data
history = model_new.fit(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

# Save the trained model
model_new.save('the_save_model.h5')
print("Model saved")

# Plot training and validation accuracy
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

# Plot training and validation loss
plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

# Show the plots
plt.show()
