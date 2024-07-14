import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

img_width, img_height = 480, 480
base_folder = "PROCESSED_Dataset"
dataset_folder = "Assets\\CROPPED_Dataset\\"
output_folder = f'{base_folder}\\PROCESSED_Dataset\\'
dataset_testing_folder = 'Assets\\RAW_TESTING_Dataset\\'
output_testing_folder = f'{base_folder}\\PROCESSED_TESTING_Dataset\\'

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_tensor = np.expand_dims(img_array, axis=0)
    img_tensor /= 255.0
    return img_tensor

# Funzione per salvare il tensore
def save_tensor(tensor, output_path):
    np.save(output_path, tensor)

# Funzione per ottenere tutte le sottocartelle
def get_all_subfolders(directory):
    leaf_subfolders = []
    for root, dirs, files in os.walk(directory):
        if not dirs:  # Verifica se la cartella corrente non ha sottocartelle
            leaf_subfolders.append(root)
    return leaf_subfolders

if not os.path.exists(output_testing_folder):    
    print("Creating output folder...")   
    os.makedirs(output_testing_folder)
    print(f'Created {output_testing_folder}...')
else:
    print("Output folder already exists.")

# Ottiene tutte le sottocartelle del dataset di testing
dataset_folder = get_all_subfolders(dataset_testing_folder)

for folder in dataset_folder:
    print(f"Processing {folder}...")
    if "STOP" in folder:
        class_name = "STOP_SIGN"
    elif "30" in folder:
        class_name = "SPEED_LIMITER_30"
    elif "60" in folder:
        class_name = "SPEED_LIMITER_60"
    elif "90" in folder:
        class_name = "SPEED_LIMITER_90"
    
    output_class_folder = os.path.join(output_testing_folder, class_name)
    
    if not os.path.exists(output_class_folder):
        os.makedirs(output_class_folder)
    
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img_tensor = preprocess_image(img_path)
        output_path = os.path.join(output_class_folder, img_name.split('.')[0] + '.npy')
        save_tensor(img_tensor, output_path)
    
    print(f"Finished processing {output_class_folder} images")

def get_file_number(file_name):
    match = re.search(r'\d+', file_name)
    if match:
        return int(match.group())
    return 0
