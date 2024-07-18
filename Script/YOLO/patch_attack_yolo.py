import numpy as np
from PIL import Image
import os
import random
import cv2


folder_path = '..\..\Assets\RAW_TESTING_Dataset\\'
output_folder = '..\..\Assets\PATCH_TESTING_Dataset4\\'

# Definire variabili globali
NET = None
OUTPUT_LAYER = None
CLASS_INDICES = None


def main():
    global NET, OUTPUT_LAYER, CLASS_INDICES

    weights_path = 'C:/Users/danie/Desktop/CARLA-weight/CARLA-weight/yolov3-tiny-obj_5000.weights'
    config_path = 'C:/Users/danie/Desktop/CARLA-weight/CARLA-weight/yolov3-tiny-obj.cfg'
    labels_path = 'C:/Users/danie/Desktop/CARLA-weight/classes.names'

    NET = cv2.dnn.readNet(weights_path, config_path)
    layer_names = NET.getLayerNames()
    OUTPUT_LAYER = [layer_names[i - 1] for i in NET.getUnconnectedOutLayers()]

    with open(labels_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    CLASS_INDICES = {
        "STOP": 11,
        "30": 0,
        "60": 1,
        "90": 2
    }

main()
def detect_signals(image_path):
    global NET, OUTPUT_LAYER, CLASS_INDICES
    range_confidence = 0
    img = cv2.imread(image_path)
    height, width, nonciserve = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    NET.setInput(blob)
    outs = NET.forward(OUTPUT_LAYER)

    class_ids = []
    confidences = []
    boxes = []
    center_x = 0
    center_y = 0
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4 and class_id in CLASS_INDICES.values():
                # calcola le coordinate del centro dell'oggetto detectato
                print(confidence)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                print(str(center_x) + " " + str(center_y))


    return center_x,center_y


#detect_signals('C:/Users/danie/PycharmProjects/DMRC/Assets/RAW_TESTING_Dataset/SPEED_LIMITER_30/30_100.png')

def get_all_subfolders(directory):
    subfolders = []
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            subfolders.append(os.path.join(root, dir))
    return subfolders


def apply_patch_filter():
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
            #random_patch = random.randint(1, 4)
            #path_filter = '..\Assets\img\patch\patch' + str(random_patch) + '.png'
            path_filter = '..\..\Assets\img\\newPatch\patch4.png'
            overlay_image = Image.open(path_filter)

            # Costruisci il percorso completo del file immagine
            image_path = os.path.join(folder_path2, image_file)

            # detect image
            coord_x,coord_y = detect_signals(image_path)

            secondaria_larghezza, secondaria_altezza = overlay_image.size
            x_pos = coord_x - secondaria_larghezza // 2
            y_pos = coord_y - secondaria_altezza // 2


            # Apri le due immagini
            base_image = Image.open(image_path)

            # Ridimensiona l'immagine di sovrapposizione
            #overlay_image = overlay_image.resize((480, 480))

            # Assicurati che l'immagine da sovrapporre abbia un canale alpha per la trasparenza
            overlay_image = overlay_image.convert("RGBA")

            # Sovrapponi l'immagine in una posizione specifica (x, y)
            base_image.paste(overlay_image, (x_pos, y_pos), overlay_image)

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



apply_patch_filter()
