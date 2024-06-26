import numpy as np
import os
import random
from scipy.ndimage import gaussian_filter

def load_tensor(file_path):
    return np.load(file_path)

def save_tensor(tensor, file_path, file_name_augmented):
    np.save(file_path + '\\' + file_name_augmented, tensor)

def augment_tensor_smoothing(tensor):

    # Gaussian smoothing
    if random.choice([True, False]):
        sigma = random.uniform(0.5, 1.5)
        tensor = gaussian_filter(tensor, sigma=sigma)

    return tensor

def augment_tensor_noise(tensor):
    # Adding Gaussian noise
    if random.choice([True, False]):
        noise = np.random.normal(0, 0.01, tensor.shape)
        tensor = tensor + noise
        tensor = np.clip(tensor, 0, 1)

    return tensor

def augment_dataset(base_dir, output_dir, categories, target_size=2000):
    for category in categories:
        category_dir = os.path.join(base_dir, category)
        files = [os.path.join(category_dir, f) for f in os.listdir(category_dir) if f.endswith('.npy')]
        count = len(files)
        output_path = os.path.join(output_dir, category)
        if not os.path.exists(os.path.join(output_dir, category)):
            os.makedirs(os.path.join(output_dir, category))
                    

        if count < target_size:
            num_to_augment = target_size - count
            i = 0
            for file_to_augment in files:
                if 'original' in file_to_augment.lower() and i in range(num_to_augment):
                    tensor = load_tensor(file_to_augment)
                    base_name_files = os.path.splitext(file_to_augment)[0]
                    augmented_tensor_smoothing = augment_tensor_smoothing(tensor)
                    file_name_augmented = base_name_files.split('\\')[-1] + f'_augmented_smoothing_{i}.npy'
                    print(file_name_augmented)
                    i +=1
                    augmented_tensor_noise = None
                    file_name_augmented2 = None
                    if i in range(num_to_augment):
                        augmented_tensor_noise = augment_tensor_noise(tensor)
                        file_name_augmented2 = base_name_files.split('\\')[-1] + f'_augmented_noise_{i}.npy'
                        print(file_name_augmented2)
                        i +=1
                    
                    save_tensor(augmented_tensor_smoothing, output_path, file_name_augmented)
                    if augmented_tensor_noise is not None or file_name_augmented2 is not None:
                        save_tensor(augmented_tensor_noise, output_path, file_name_augmented2)   

def main():
    base_dir = 'C:\\Users\\rocco\\OneDrive\\Desktop\\DMRC-1\\PROCESSED_Dataset\\PROCESSED_Dataset'
    output_dir = 'C:\\Users\\rocco\\OneDrive\\Desktop\\DMRC-1\\PROCESSED_Dataset\\augmented_dataset'
    categories = ['SPEED_LIMITER_30', 'SPEED_LIMITER_60', 'SPEED_LIMITER_90', 'STOP_SIGN']
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    augment_dataset(base_dir, output_dir, categories, target_size=2000)

if __name__ == "__main__":
    main()
