import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import datetime

learning_rate = 0

def create_alexnet_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(96, kernel_size=(11, 11), strides=4, padding='valid', activation='relu', input_shape=input_shape, kernel_initializer='he_normal'),
        MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'),
        Conv2D(256, kernel_size=(5, 5), strides=1, padding='same', activation='relu', kernel_initializer='he_normal'),
        MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'),
        Conv2D(384, kernel_size=(3, 3), strides=1, padding='same', activation='relu', kernel_initializer='he_normal'),
        Conv2D(384, kernel_size=(3, 3), strides=1, padding='same', activation='relu', kernel_initializer='he_normal'),
        Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation='relu', kernel_initializer='he_normal'),
        MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_class_weights(directory):
    classes = os.listdir(directory)
    num_samples_per_class = {}
    for cls in classes:
        path = os.path.join(directory, cls)
        count = 0
        for root, dirs, files in os.walk(path):
            count += len([file for file in files if file.endswith('.jpg') or file.endswith('.png')])
        num_samples_per_class[cls] = count
    total_samples = sum(num_samples_per_class.values())
    class_weights = {i: total_samples / (len(classes) * num_samples_per_class[cls]) for i, cls in enumerate(classes)}
    return class_weights, num_samples_per_class


base_dir = 'C:\\Users\\rocco\\OneDrive\\Desktop\\DMRC-1\\Assets\\CROPPED_Dataset'
batch_size = 32
input_shape = (227, 227, 3)
num_classes = 4

class_weights, num_samples_per_class = get_class_weights(base_dir)
augmentation_factor = np.mean([max(num_samples_per_class.values()) / num for num in num_samples_per_class.values()])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1 * augmentation_factor,
    height_shift_range=0.1 * augmentation_factor,
    shear_range=0.1,
    zoom_range=0.1,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

model = create_alexnet_model(input_shape, num_classes)
output_dir = 'C:\\Users\\rocco\\OneDrive\\Desktop\\OUTPUT_Dataset_Image'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_checkpoint = ModelCheckpoint(os.path.join(output_dir, 'best_model_alexnet.keras'), save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

os.mkdir('./logs')
os.mkdir('./logs/fit')

log_dir="./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
callback_list = [tensorboard_callback]


history = model.fit(
    train_generator,
    epochs=90,
    validation_data=val_generator,
    callbacks=[model_checkpoint, early_stopping, callback_list]
)

model.save(os.path.join(output_dir, 'final_model_alexnet.h5'))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig(os.path.join(output_dir, 'performance_plots.png'))
plt.show()
