import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_files_in_directory(base_dir, base_augmented_dir, categories):
    data = []
    weather_conditions = ['clearnoon', 'clearnight', 'fognoon', 'fognight', 
                          'hardrainnoon', 'hardrainnight', 'midrainnoon', 'midrainnight']
    
    for category in categories:
        category_dir = os.path.join(base_dir, category)
        for root, dirs, files in os.walk(category_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_type = 'origin' if 'original' in file.lower() else 'cropped'
                condition = 'unknown'
                for weather in weather_conditions:
                    if weather in file.lower():
                        condition = weather
                        break
                data.append({'Category': category, 'File_Type': file_type, 'Condition': condition, 'File_Path': file_path})
        
        augmented_dir = os.path.join(base_augmented_dir, category)
        for root, dirs, files in os.walk(augmented_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_type = 'origin_augmented' if 'original' in file.lower() else 'cropped'
                condition = 'unknown'
                for weather in weather_conditions:
                    if weather in file.lower():
                        condition = weather
                        break
                data.append({'Category': category, 'File_Type': file_type, 'Condition': condition, 'File_Path': file_path})
    
    return pd.DataFrame(data)

def visualize_data(file_path):
    df = pd.read_csv(file_path)
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Category')
    plt.title('Distribution of Files per Category')
    plt.xlabel('Category')
    plt.ylabel('Number of Files')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('figure_distribution\\category_distribution.jpg')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Condition')
    plt.title('Distribution of Files per Condition')
    plt.xlabel('Condition')
    plt.ylabel('Number of Files')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('figure_distribution\\condition_distribution.jpg')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='File_Type')
    plt.title('Distribution of Files per Type')
    plt.xlabel('File Type')
    plt.ylabel('Number of Files')
    plt.tight_layout()
    plt.savefig('figure_distribution\\file_type_distribution.jpg')
    plt.show()
    
    plt.figure(figsize=(14, 8))
    sns.countplot(data=df, x='Category', hue='Condition')
    plt.title('Distribution of Files per Category and Condition')
    plt.xlabel('Category')
    plt.ylabel('Number of Files')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('figure_distribution\\category_condition_distribution.jpg')
    plt.show()

def main():
    base_dir = "C:\\Users\\rocco\\OneDrive\\Desktop\\PROCESSED_Dataset\\PROCESSED_Dataset"
    base_augmented_dir = 'C:\\Users\\rocco\\OneDrive\\Desktop\\PROCESSED_Dataset\\augmented_dataset'
    categories = ['SPEED_LIMITER_30', 'SPEED_LIMITER_60', 'SPEED_LIMITER_90', 'STOP_SIGN']
    
    df = analyze_files_in_directory(base_dir, base_augmented_dir, categories)
    print(df.head())
    
    # Save the composition of files in a CSV
    output_path = 'C:\\Users\\rocco\\OneDrive\\Desktop\\DMRC-1\\PROCESSED_Dataset\\file_composition.csv'
    df.to_csv(output_path, index=False)
    print(f'File composition saved to {output_path}')
    visualize_data(output_path)

if __name__ == "__main__":
    main()
