import os
import random
from sklearn.model_selection import train_test_split

def get_files_from_directory(directory):
    """
    Get a list of all files in a directory and its subdirectories.
    """
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def split_data(files, train_size=0.7, val_size=0.15, test_size=0.15):
    """
    Split the files into train, validation, and test sets.
    """
    train_files, temp_files = train_test_split(files, train_size=train_size)
    val_files, test_files = train_test_split(temp_files, test_size=test_size/(val_size+test_size))
    return train_files, val_files, test_files

def save_files_to_directory(files, output_dir):
    """
    Save files to the specified directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for file in files:
        os.symlink(file, os.path.join(output_dir, os.path.basename(file)))

def main():
    base_dir = '/mnt/data/PROCESSED_Dataset/PROCESSED_Dataset'
    categories = ['SPEED_LIMITER_30', 'SPEED_LIMITER_60', 'SPEED_LIMITER_90', 'STOP_SIGN']
    output_base_dir = '/mnt/data/PROCESSED_Dataset/OUTPUT_Dataset'

    for category in categories:
        category_dir = os.path.join(base_dir, category)
        files = get_files_from_directory(category_dir)
        
        train_files, val_files, test_files = split_data(files)
        
        save_files_to_directory(train_files, os.path.join(output_base_dir, 'train', category))
        save_files_to_directory(val_files, os.path.join(output_base_dir, 'val', category))
        save_files_to_directory(test_files, os.path.join(output_base_dir, 'test', category))

if __name__ == "__main__":
    main()
