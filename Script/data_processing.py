import random

import os
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils import image_utils as image

#Global Variables
img_width, img_height = 480, 480

#PATHS
base_folder = "..\..\Assets"
# folders = [f'{base_path}\CROPPED_Dataset\SPEED_LIMITER_30',f'{base_path}\CROPPED_Dataset\SPEED_LIMITER_60',f'{base_path}\CROPPED_Dataset\SPEED_LIMITER_90',f'{base_path}\CROPPED_Dataset\STOP_SIGN',]
dataset_folder = f"{base_folder}\CROPPED_Dataset\\"
output_folder = f'{base_folder}\PROCESSED_Dataset\\'

#IMAGE PRE-PROCESSING
def preprocess_image(image_path):
    #Loading Image
    img = image.load_img(image_path, target_size=(img_width, img_height))

    #Conversion to a Numpy array
    img_array = image.img_to_array(img)

    #Image Normalization
    img_tensor = np.expand_dims(img_array, axis=0)
    img_tensor /= 255.0

    return img_tensor

#Save the Tensor in a npy file into the "output_path"
def save_tensor(tensor, output_path):
    np.save(output_path , tensor)

def get_all_subfolders(directory):
    subfolders = []
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            subfolders.append(os.path.join(root, dir))
    return subfolders
    
if not os.path.exists(output_folder):    
    print("Creating output folder...")   
    os.makedirs(output_folder)

dataset_folder = [element for element in get_all_subfolders(dataset_folder) if 'Padding' in element]

for folder in dataset_folder:
    if "STOP" in folder:
        class_name = "STOP_SIGN"
    elif "30" in folder:
        class_name = "SPEED_LIMITER_30"
    elif "60" in folder:
        class_name = "SPEED_LIMITER_60"
    elif "90" in folder:
        class_name = "SPEED_LIMITER_90"    
    
    output_class_folder = os.path.join(output_folder, class_name)
        
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