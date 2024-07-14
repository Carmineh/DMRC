from PIL import Image, ImageDraw, ImageFilter
import os
import random

folder_path = '..\Assets\RAW_TESTING_Dataset\\'
output_folder = '..\Assets\PATCH_TESTING_Dataset\\'


# prova in locale
# folder_path = 'RAW_TESTING_Dataset'
# output_folder = 'GLASS_TESTING_Dataset'

def get_all_subfolders(directory):
    subfolders = []
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            subfolders.append(os.path.join(root, dir))
    return subfolders


def apply_glass_filter():
    dataset_folder = get_all_subfolders(folder_path)
    for folder in dataset_folder:
        if "STOP" in folder:
            class_name = "STOP_SIGN"
        elif "30" in folder:
            class_name = "SPEED_LIMITER_30"
        elif "60" in folder:
            class_name = "SPEED_LIMITER_60"
        elif "90" in folder:
            class_name = "SPEED_LIMITER_90"

        folder_path2 = os.path.join(folder_path, class_name)

        # Crea la cartella di output se non esiste
        os.makedirs(output_folder, exist_ok=True)

        # Ottieni la lista di tutti i file nella cartella
        file_list = os.listdir(folder_path2)



        for image_file in file_list:
            # scelta random del filter
            random_patch = random.randint(1, 4)
            path_filter = '..\Assets\img\patch\patch' + str(random_patch) + '.png'
            overlay_image = Image.open(path_filter)

            # Costruisci il percorso completo del file immagine
            image_path = os.path.join(folder_path2, image_file)

            # Apri le due immagini
            base_image = Image.open(image_path)

            # Ridimensiona l'immagine di sovrapposizione
            overlay_image = overlay_image.resize((480, 480))

            # Assicurati che l'immagine da sovrapporre abbia un canale alpha per la trasparenza
            overlay_image = overlay_image.convert("RGBA")

            # Sovrapponi l'immagine in una posizione specifica (x, y)
            base_image.paste(overlay_image, (50, 50), overlay_image)

            output_path = os.path.join(output_folder, class_name)
            # Crea la cartella di output se non esiste
            os.makedirs(output_path, exist_ok=True)

            # Costruisci il percorso completo per il file di output
            output_path2 = os.path.join(output_path, image_file)

            # Salva il risultato
            base_image.save(output_path2)

            base_image.close()

            # Visualizza il risultato
            # base_image.show()


apply_glass_filter()