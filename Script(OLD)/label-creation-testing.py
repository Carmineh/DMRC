import os
import random
import numpy as np

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
    base_dir = 'C:\\Users\\rocco\\OneDrive\\Desktop\\PROCESSED_Dataset'
    categories = {
        'SPEED_LIMITER_30': 0,
        'SPEED_LIMITER_60': 1,
        'SPEED_LIMITER_90': 2,
        'STOP_SIGN': 3
    }

    test_tensors = []
    test_labels = []
    
    for category, label in categories.items():
        category_dir_processed = os.path.join(base_dir, 'PROCESSED_TESTING_Dataset', category)
        
        files_processed = get_files_from_directory(category_dir_processed)
        
        # Campionatura casuale dei file per il testing
        testing_files = random.sample(files_processed, len(files_processed))
        
        test_tensors_category, test_labels_category = create_dataset_and_labels(testing_files, label)
        test_tensors.extend(test_tensors_category)
        test_labels.extend(test_labels_category)

    # Creazione array numpy
    test_tensors = np.array(test_tensors)
    test_labels = np.array(test_labels)

    print("Testing set tensors shape:", test_tensors.shape)
    print("Testing set labels shape:", test_labels.shape)

    # Directory di output
    output_dir = 'C:\\Users\\rocco\\OneDrive\\Desktop\\OUTPUT_Dataset'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Salva gli array numpy in file
    np.save(os.path.join(output_dir, 'test_tensors.npy'), test_tensors)
    np.save(os.path.join(output_dir, 'test_labels.npy'), test_labels)

    print("Testing set tensors salvato")
    print("Testing set labels salvato")

if __name__ == "__main__":
    main()
