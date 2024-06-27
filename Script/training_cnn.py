import os
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras import regularizers

# Definisci un generatore per caricare i dati in modo incrementale
def data_generator(tensors_file, labels_file, batch_size=32):
    tensors = np.load(tensors_file, mmap_mode='r')  # Utilizza mmap_mode='r' per la lettura incrementale
    labels = np.load(labels_file)
    num_samples = len(tensors)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        yield tensors[batch_indices], labels[batch_indices]

# Trasforma le etichette in formato one-hot durante il preprocessamento
def preprocess(tensors, labels, num_classes):
    labels = tf.one_hot(labels, depth=num_classes)
    return tensors, labels

def main():
    output_dir = 'C:\\Users\\rocco\\OneDrive\\Desktop\\DMRC-1\\OUTPUT_Dataset'

    # Creazione del dataset TensorFlow dai file `.npy` utilizzando il generatore
    train_tensors_file = os.path.join(output_dir, 'train_tensors.npy')
    train_labels_file = os.path.join(output_dir, 'train_labels.npy')
    val_tensors_file = os.path.join(output_dir, 'val_tensors.npy')
    val_labels_file = os.path.join(output_dir, 'val_labels.npy')

    # Calcola il numero di classi dalle etichette
    num_classes = len(np.unique(np.load(train_labels_file)))

    batch_size = 32  # Definisci la dimensione del batch

    # Creazione dei dataset TensorFlow
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

    # Applica la funzione preprocess passando il numero di classi
    train_ds = train_ds.map(lambda x, y: preprocess(x, y, num_classes)).repeat()
    val_ds = val_ds.map(lambda x, y: preprocess(x, y, num_classes)).repeat()

    # Prefetch per migliorare le prestazioni
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)

    # Ottieni il numero totale di campioni per calcolare steps_per_epoch e validation_steps
    num_train_samples = len(np.load(train_tensors_file))
    num_val_samples = len(np.load(val_tensors_file))

    # Calcola il numero totale di batch per epoca e per la validazione
    steps_per_epoch = np.ceil(num_train_samples / batch_size).astype(int)
    validation_steps = np.ceil(num_val_samples / batch_size).astype(int)

    print("Training set batches created with batch size:", batch_size)
    print("Validation set batches created with batch size:", batch_size)
    print("Steps per epoch:", steps_per_epoch)
    print("Validation steps:", validation_steps)

    # Verifica dei dataset creati
    for images, labels in train_ds.take(1):
        print("Batch di training - Immagini:", images.shape, "Etichette:", labels.shape)
    for images, labels in val_ds.take(1):
        print("Batch di validazione - Immagini:", images.shape, "Etichette:", labels.shape)

    # Definizione del modello
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(480, 480, 3)),
        MaxPool2D(pool_size=(2, 2)),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),
        Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),
        Flatten(),
        Dense(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dropout(0.5),
        Dense(units=64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dropout(0.5),
        Dense(units=32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dense(units=4, activation='softmax')
    ])

    # Compilazione del modello
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Addestramento del modello
    model.fit(train_ds, validation_data=val_ds, epochs=10, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

    model.save( output_dir + '\\model.h5')
    model.save(output_dir + '\\my_model.keras')

if __name__ == "__main__":
    main()
