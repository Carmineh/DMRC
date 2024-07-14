import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, Add, Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Configura TensorFlow per utilizzare la GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Configura TensorFlow per utilizzare la prima GPU disponibile
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU is available and configured")
    except RuntimeError as e:
        print(e)
else:
    print("GPU is not available")

# Funzione per il blocco MBConv
def MBConvBlock(inputs, filters, kernel_size, strides, expansion_factor):
    x = inputs
    x = Conv2D(filters * expansion_factor, (1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = DepthwiseConv2D(kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters, (1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    if strides == 1 and inputs.shape[-1] == filters:
        x = Add()([inputs, x])
    return x

# Costruzione del modello basato sull'architettura fornita
def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    # Primo livello di convoluzione
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Block 1
    x = MBConvBlock(x, filters=16, kernel_size=3, strides=1, expansion_factor=1)

    # Block 2
    x = MBConvBlock(x, filters=24, kernel_size=3, strides=2, expansion_factor=6)
    x = MBConvBlock(x, filters=24, kernel_size=3, strides=1, expansion_factor=6)

    # Block 3
    x = MBConvBlock(x, filters=40, kernel_size=5, strides=2, expansion_factor=6)
    x = MBConvBlock(x, filters=40, kernel_size=5, strides=1, expansion_factor=6)

    # Block 4
    x = MBConvBlock(x, filters=80, kernel_size=3, strides=2, expansion_factor=6)
    x = MBConvBlock(x, filters=80, kernel_size=3, strides=1, expansion_factor=6)
    x = MBConvBlock(x, filters=80, kernel_size=3, strides=1, expansion_factor=6)

    # Block 5
    x = MBConvBlock(x, filters=112, kernel_size=5, strides=1, expansion_factor=6)
    x = MBConvBlock(x, filters=112, kernel_size=5, strides=1, expansion_factor=6)

    # Block 6
    x = MBConvBlock(x, filters=192, kernel_size=5, strides=2, expansion_factor=6)
    x = MBConvBlock(x, filters=192, kernel_size=5, strides=1, expansion_factor=6)
    x = MBConvBlock(x, filters=192, kernel_size=5, strides=1, expansion_factor=6)
    x = MBConvBlock(x, filters=192, kernel_size=5, strides=1, expansion_factor=6)
    x = MBConvBlock(x, filters=192, kernel_size=5, strides=1, expansion_factor=6)
    x = MBConvBlock(x, filters=192, kernel_size=5, strides=1, expansion_factor=6)

    # Block 7
    x = MBConvBlock(x, filters=320, kernel_size=3, strides=1, expansion_factor=6)

    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, x)
    return model

# Carica i dati in maniera incrementale
def data_generator(tensors_file, labels_file, batch_size=32):
    tensors = np.load(tensors_file, mmap_mode='r')  # mmap_mode='r' è la lettura incrementale
    labels = np.load(labels_file)
    num_samples = len(tensors)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        yield tensors[batch_indices], labels[batch_indices]

# Trasforma le etichette in formato one-hot e ridimensiona le immagini
def preprocess(tensors, labels, num_classes, target_size):
    tensors = tf.image.resize(tensors, target_size)
    labels = tf.one_hot(labels, depth=num_classes)
    return tensors, labels

def main():
    output_dir = 'C:\\Users\\rocco\\OneDrive\\Desktop\\OUTPUT_Dataset_Nonoriginal'

    # Carico i file .npy
    train_tensors_file = os.path.join(output_dir, 'train_tensors.npy')
    train_labels_file = os.path.join(output_dir, 'train_labels.npy')
    val_tensors_file = os.path.join(output_dir, 'val_tensors.npy')
    val_labels_file = os.path.join(output_dir, 'val_labels.npy')

    # Calcola il numero di classi dalle etichette
    num_classes = len(np.unique(np.load(train_labels_file)))

    batch_size = 32  # Dimensione del batch

    train_ds = tf.data.Dataset.from_generator(
        lambda: data_generator(train_tensors_file, train_labels_file, batch_size),
        output_signature=(
            tf.TensorSpec(shape=(None, 480, 480, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int64))
    )

    val_ds = tf.data.Dataset.from_generator(
        lambda: data_generator(val_tensors_file, val_labels_file, batch_size),
        output_signature=(
            tf.TensorSpec(shape=(None, 480, 480, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int64))
    )

    target_size = (224, 224)
    train_ds = train_ds.map(lambda x, y: preprocess(x, y, num_classes, target_size)).repeat()
    val_ds = val_ds.map(lambda x, y: preprocess(x, y, num_classes, target_size)).repeat()

    # Prefetch per migliorare le prestazioni
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)

    num_train_samples = len(np.load(train_tensors_file))
    num_val_samples = len(np.load(val_tensors_file))

    steps_per_epoch = np.ceil(num_train_samples / batch_size).astype(int)
    validation_steps = np.ceil(num_val_samples / batch_size).astype(int)

    print("Training set batches created with batch size:", batch_size)
    print("Validation set batches created with batch size:", batch_size)
    print("Steps per epoch:", steps_per_epoch)
    print("Validation steps:", validation_steps)

    # Verifico i dati
    for images, labels in train_ds.take(1):
        print("Batch di training - Immagini:", images.shape, "Etichette:", labels.shape)
    for images, labels in val_ds.take(1):
        print("Batch di validazione - Immagini:", images.shape, "Etichette:", labels.shape)

    # Costruzione del modello
    input_shape = (224, 224, 3)
    model = build_model(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Per farlo fermare prima se non si migliora
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(os.path.join(output_dir, 'best_model.keras'), save_best_only=True)

    history = model.fit(train_ds, validation_data=val_ds, epochs=20, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, callbacks=[early_stopping, model_checkpoint])

    model.save(os.path.join(output_dir, 'model.keras'))

    # Grafico della perdita e dell'accuratezza
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
    plt.savefig(os.path.join(output_dir, 'training_validation_plots.png'))
    plt.show()

if __name__ == "__main__":
    main()
