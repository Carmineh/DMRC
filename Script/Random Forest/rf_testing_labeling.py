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
    base_dir = "..\..\Assets\PROCESSED_TESTING_Dataset"
    categories = {
        "SPEED_LIMITER_30": 0,
        "SPEED_LIMITER_60": 1,
        "SPEED_LIMITER_90": 2,
        "STOP_SIGN": 3,
    }

    test_tensors = []
    test_labels = []

    for category, label in categories.items():
        category_dir_processed = os.path.join(base_dir, category)

        files_processed = get_files_from_directory(category_dir_processed)

        total_files = len(files_processed)
        print(f"{category}: {total_files} files")

        test_files = random.sample(files_processed, total_files)

        test_tensors_temp, test_labels_temp = create_dataset_and_labels(
            test_files, label
        )
        test_tensors.extend(test_tensors_temp)
        test_labels.extend(test_labels_temp)

        print(len(test_tensors))
        print(len(test_labels))

    test_tensors = np.array(test_tensors)
    test_labels = np.array(test_labels)

    print("Test set tensors shape:", test_tensors.shape)
    print("Test set labels shape:", test_labels.shape)

    # Directory di output
    output_dir = "..\Assets\OUTPUT_TESTING_Dataset"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Salva gli array numpy in file
    np.save(os.path.join(output_dir, "test_tensors.npy"), test_tensors)
    print("Test tensors saved in", os.path.join(output_dir, "test_tensors.npy"))

    np.save(os.path.join(output_dir, "test_labels.npy"), test_labels)
    print("Test labels saved in", os.path.join(output_dir, "test_labels.npy"))


if __name__ == "__main__":
    main()
