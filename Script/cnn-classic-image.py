import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt

learning_rate = 0

def create_efficientnet(input_shape, num_classes):
    base_model = EfficientNetB0(include_top=False, input_shape=input_shape, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
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

def only_return_learning_rate(epoch, lr):
    global learning_rate
    print(learning_rate)
    return learning_rate

class AdaptiveLearningRateAndClassWeights(Callback):
    def __init__(self, initial_class_weights, val_generator, patience=2, max_weight_factor=2.0, warmup_epochs=3):
        super().__init__()
        self.patience = patience
        self.wait = 0
        self.best_weights = None
        self.best_loss = float('inf')
        self.initial_class_weights = initial_class_weights.copy()
        self.class_weights = initial_class_weights.copy()
        self.val_generator = val_generator
        self.max_weight_factor = max_weight_factor
        self.best_val_loss = 0
        self.best_val_accuracy = 0
        self.lr_decrease_count = 0
        self.warmup_epochs = warmup_epochs
        self.epoch_count = 0
        
    def on_train_begin(self, logs=None):
        global learning_rate
        self.base_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        self.previous_lr = self.base_lr
        learning_rate = self.base_lr
        print(f"Starting training with initial learning rate: {self.base_lr}")
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_count += 1
        
    def on_epoch_end(self, epoch, logs=None):

        current_val_loss = logs.get('val_loss')
        if (current_val_loss >= self.best_val_loss ):
            self.best_val_loss = current_val_loss
        current_val_accuracy = logs.get('val_accuracy', 0)  

        if current_val_loss < self.best_loss:
            self.best_loss = current_val_loss
            self.wait = 0
            self.best_weights = self.model.get_weights() 
        else:
            self.wait += 1

        if self.wait >= self.patience:
            if current_val_accuracy < self.best_val_accuracy - 0.05:
                self.model.set_weights(self.best_weights)
                print("Restored best weights due to performance degradation.")
            self.adjust_class_weights(logs)
            self.wait = 0
            print(f"Adjusted class weights after {self.patience} epochs without improvement.")

        self.adjust_learning_rate(current_val_loss, current_val_accuracy)

    def adjust_class_weights(self, logs):
        errors_by_class = self.evaluate_errors_by_class()
        sorted_classes = sorted(errors_by_class.items(), key=lambda item: item[1], reverse=True)
        for i, (cls, error) in enumerate(sorted_classes):
            adjustment = 1.0 + (len(sorted_classes) - i - 1) * 0.1  # Increasing weight adjustment
            self.class_weights[cls] = min(max(self.initial_class_weights[cls] * adjustment, self.initial_class_weights[cls] / self.max_weight_factor), self.initial_class_weights[cls] * self.max_weight_factor)
        self.model.class_weight = self.class_weights
        print(f"Adjusted class weights: {self.class_weights}")

    def adjust_learning_rate(self, current_val_loss, current_val_accuracy):
        global learning_rate
        new_lr = self.previous_lr

        if self.epoch_count <= self.warmup_epochs:
            new_lr = self.base_lr
        else:
            loss_improvement = (self.best_val_loss - current_val_loss) / max(self.best_val_loss, 1e-6)
            accuracy_improvement = current_val_accuracy - self.best_val_accuracy

            if loss_improvement > 0:
                new_lr *= (1 + min(loss_improvement, 0.1))
                self.lr_decrease_count = 0
            elif self.lr_decrease_count < 3:
                new_lr *= 0.95
                self.lr_decrease_count += 1

            if accuracy_improvement > 0:
                new_lr *= 1.02
                self.lr_decrease_count = 0
            elif self.lr_decrease_count < 3:
                new_lr *= 0.98
                self.lr_decrease_count += 1

            if current_val_accuracy > 0.80:
                excess_accuracy = current_val_accuracy - 0.80
                decrease_factor = 1 - min(excess_accuracy * 0.5, 0.1)
                new_lr *= decrease_factor

            new_lr = max(min(new_lr, self.base_lr * 1.5), self.base_lr * 0.1)

        learning_rate = new_lr
        if new_lr != self.previous_lr:
            self.previous_lr = new_lr
            print(f"Adjusted learning rate to {new_lr:.6f} due to validation performance changes.")


    def evaluate_errors_by_class(self):
        error_counts = {i: 0 for i in range(len(self.initial_class_weights))}
        total_counts = {i: 0 for i in range(len(self.initial_class_weights))}

        for batch in self.val_generator:
            x_val, y_val = batch
            y_pred = np.argmax(self.model.predict(x_val), axis=-1)
            y_true = np.argmax(y_val, axis=-1)
            for i in range(len(y_true)):
                total_counts[y_true[i]] += 1
                if y_true[i] != y_pred[i]:
                    error_counts[y_true[i]] += 1
            if len(y_true) < self.val_generator.batch_size:
                break

        errors_by_class = {cls: (error_counts[cls] / total_counts[cls]) if total_counts[cls] > 0 else 1 for cls in error_counts}
        print(f"Errors by class: {errors_by_class}")
        return errors_by_class

base_dir = 'C:\\Users\\rocco\\OneDrive\\Desktop\\DMRC-1\\Assets\\CROPPED_Dataset'
batch_size = 32
input_shape = (224, 224, 3)
num_classes = 4

class_weights, num_samples_per_class = get_class_weights(base_dir)
augmentation_factor = np.mean([max(num_samples_per_class.values()) / num for num in num_samples_per_class.values()])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2 * augmentation_factor,
    height_shift_range=0.2 * augmentation_factor,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
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

model = create_efficientnet(input_shape, num_classes)
output_dir = 'C:\\Users\\rocco\\OneDrive\\Desktop\\OUTPUT_Dataset_Image'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_checkpoint = ModelCheckpoint(os.path.join(output_dir, 'best_model.keras'), save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
adaptive_lr_and_weights = AdaptiveLearningRateAndClassWeights(initial_class_weights=class_weights, val_generator=val_generator, patience=2)
lr_scheduler = LearningRateScheduler(only_return_learning_rate)


history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=[model_checkpoint, early_stopping, adaptive_lr_and_weights, lr_scheduler]
)

model.save(os.path.join(output_dir, 'final_model.h5'))

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
