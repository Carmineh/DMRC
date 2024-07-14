import shutil
import numpy as np
import os
import random
from scipy.ndimage import rotate
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_tensor(file_path):
    tensor = np.load(file_path)
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(axis=0)
    return tensor

def save_tensor(tensor, file_path, file_name):
    np.save(os.path.join(file_path, file_name), tensor)

def denormalize_tensor(tensor):
    return tensor * 255.0

def normalize_tensor(tensor):
    return tensor / 255.0

def augment_tensor_translation(tensor):
    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1)
    tensor = np.expand_dims(tensor, 0)
    it = datagen.flow(tensor, batch_size=1)
    return next(it)[0]

def augment_tensor_zoom(tensor):
    datagen = ImageDataGenerator(zoom_range=0.1)
    tensor = np.expand_dims(tensor, 0)
    it = datagen.flow(tensor, batch_size=1)
    return next(it)[0]

def augment_tensor_rotation(tensor):
    angle = random.uniform(-15, 15)
    return rotate(tensor, angle, reshape=False)

def augment_tensor_noise(tensor):
    noise = np.random.normal(0, 0.01, tensor.shape)
    tensor = tensor + noise
    return np.clip(tensor, 0, 255)

def augment_dataset(base_dir, output_dir, categories, target_size):
    augmentations = [augment_tensor_translation, augment_tensor_zoom, augment_tensor_rotation, augment_tensor_noise]
    
    for category in categories:
        category_dir = os.path.join(base_dir, category)
        files = [os.path.join(category_dir, f) for f in os.listdir(category_dir) if f.endswith('.npy') and 'original' in f]
        count = len(files)
        print(count)
        output_path = os.path.join(output_dir, category)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if count < target_size:

            num_to_augment = target_size - count
            augmented_count = 0
            original_files = [f for f in files if 'original' in f.lower()]
            used_files = set()
            used_augmentations = {}

            while augmented_count < num_to_augment and original_files:
                file = random.choice(original_files)
                if file not in used_files:
                    used_files.add(file)
                    used_augmentations[file] = set()

                available_augmentations = list(set(augmentations) - used_augmentations[file])
                if not available_augmentations:
                    continue

                aug_func = random.choice(available_augmentations)
                used_augmentations[file].add(aug_func)
                tensor = load_tensor(file)
                tensor = denormalize_tensor(tensor)

                augmented_tensor = aug_func(tensor)
                augmented_tensor = normalize_tensor(augmented_tensor)
                
                base_name = os.path.splitext(os.path.basename(file))[0]
                file_name_augmented = f"{base_name}_augmented_{aug_func.__name__}_{augmented_count}.npy"
                save_tensor(augmented_tensor, output_path, file_name_augmented)
                
                augmented_count += 1

def main():
    base_dir = "C:\\Users\\rocco\\OneDrive\\Desktop\\PROCESSED_Dataset\\PROCESSED_Dataset"
    output_dir = 'C:\\Users\\rocco\\OneDrive\\Desktop\\PROCESSED_Dataset\\augmented_dataset_original'
    categories = ['SPEED_LIMITER_30', 'SPEED_LIMITER_60', 'SPEED_LIMITER_90', 'STOP_SIGN']
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    
    augment_dataset(base_dir, output_dir, categories, target_size=1000)

if __name__ == "__main__":
    main()
