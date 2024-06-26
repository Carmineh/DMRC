import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import tensorflow as tf

def get_files_from_directory(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def load_tensor(file_path):
    tensor = np.load(file_path)
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(axis=0)  # Rimuove la dimensione extra se presente
    return tensor

def create_dataset_and_labels(files, label):
    tensors = []
    labels = []
    for file in files:
        tensor = load_tensor(file)
        tensors.append(tensor)
        labels.append(label)
    return tensors, labels

def main():
    base_dir = 'C:\\Users\\rocco\\OneDrive\\Desktop\\DMRC-1\\PROCESSED_Dataset'
    categories = {
        'SPEED_LIMITER_30': 0,
        'SPEED_LIMITER_60': 1,
        'SPEED_LIMITER_90': 2,
        'STOP_SIGN': 3
    }

    train_tensors = []
    train_labels = []
    val_tensors = []
    val_labels = []
    
    for category, label in categories.items():
        category_dir_augmented = os.path.join(base_dir, 'augmented_dataset', category)
        category_dir_processed = os.path.join(base_dir, 'PROCESSED_Dataset', category)
        
        files_augmented = get_files_from_directory(category_dir_augmented)
        files_processed = get_files_from_directory(category_dir_processed)
        
        original_files = [f for f in files_processed if 'original' in f.lower()]
        non_original_files = [f for f in files_processed if 'original' not in f.lower()]
        
        # Calculate the total number of files
        total_files = len(original_files) + len(non_original_files) + len(files_augmented)
        number_of_validation_set = int(total_files * 0.10)
        # Split original files into training and validation sets
        val_files_original = random.sample(original_files, number_of_validation_set)
        val_files_nonoriginal = random.sample(non_original_files, number_of_validation_set)
        val_files = val_files_original + val_files_nonoriginal
        train_files_partial = [f for f in original_files + non_original_files if f not in val_files]
        
        train_files = train_files_partial + files_augmented
        
        train_tensors_category, train_labels_category = create_dataset_and_labels(train_files, label)
        train_tensors.extend(train_tensors_category)
        train_labels.extend(train_labels_category)
        
        val_tensors_category, val_labels_category = create_dataset_and_labels(val_files, label)
        val_tensors.extend(val_tensors_category)
        val_labels.extend(val_labels_category)

    # Convert lists to numpy arrays
    train_tensors = np.array(train_tensors)
    train_labels = np.array(train_labels)
    val_tensors = np.array(val_tensors)
    val_labels = np.array(val_labels)

    print("Training set tensors shape:", train_tensors.shape)
    print("Training set labels shape:", train_labels.shape)
    print("Validation set tensors shape:", val_tensors.shape)
    print("Validation set labels shape:", val_labels.shape)

    # Directory di output
    output_dir = 'C:\\Users\\rocco\\OneDrive\\Desktop\\DMRC-1\\OUTPUT_Dataset'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Salva gli array numpy in file
    np.save(os.path.join(output_dir, 'train_tensors.npy'), train_tensors)
    np.save(os.path.join(output_dir, 'train_labels.npy'), train_labels)
    np.save(os.path.join(output_dir, 'val_tensors.npy'), val_tensors)
    np.save(os.path.join(output_dir, 'val_labels.npy'), val_labels)

    print("Training set tensors saved to train_tensors.npy")
    print("Training set labels saved to train_labels.npy")
    print("Validation set tensors saved to val_tensors.npy")
    print("Validation set labels saved to val_labels.npy")

if __name__ == "__main__":
    main()
