import shutil
import cv2
import numpy as np
import os

# Definire variabili globali
NET = None
OUTPUT_LAYER = None
CLASS_INDICES = None

def main():
    global NET, OUTPUT_LAYER, CLASS_INDICES

    weights_path = 'C:/Users/rocco/OneDrive/Desktop/CARLA-weight/CARLA-weight/yolov3-tiny-obj_5000.weights'
    config_path = 'C:/Users/rocco/OneDrive/Desktop/CARLA-weight/CARLA-weight/yolov3-tiny-obj.cfg'
    labels_path = 'C:/Users/rocco/OneDrive/Desktop/CARLA-weight/classes.names'
    padding = [0, 20, 40, 60, 80]

    # YOLO
    NET = cv2.dnn.readNet(weights_path, config_path)
    layer_names = NET.getLayerNames()
    OUTPUT_LAYER = [layer_names[i - 1] for i in NET.getUnconnectedOutLayers()]

    # Load COCO labels
    with open(labels_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    CLASS_INDICES = {
        "STOP": 11,  
        "30": 0,    
        "60": 1,    
        "90": 2     
    }
    
    main_input_folder = 'Assets/RAW_Dataset/'
    main_output_folder = 'cropped/'
    leaf_dirs = get_all_leaf_directories(main_input_folder)
    for single_padding in padding:
        crop_image_principal(main_input_folder, main_output_folder, leaf_dirs, single_padding)

def get_all_leaf_directories(folder_path):
    leaf_directories = []

    for root, dirs, files in os.walk(folder_path):
        if not dirs:
            relative_path = os.path.relpath(root, folder_path)
            leaf_directories.append(relative_path)

    return leaf_directories

def crop_image_principal(main_input_folder, main_output_folder, leaf_dirs, single_padding):
    for leaf_dir in leaf_dirs:
        input_folder = os.path.join(main_input_folder, leaf_dir)
        output_folder = os.path.join(main_output_folder, leaf_dir + "\\Padding" + str(single_padding))

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        else:
            shutil.rmtree(output_folder)
            os.makedirs(output_folder)

        # Itera su tutte le immagini nella cartella di input
        for filename in os.listdir(input_folder):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                image_path = os.path.join(input_folder, filename)
                detect_and_crop(image_path, output_folder, single_padding)

#gi passiamo il path dell'immagine, la cartella di output, il grado di affinità della detection e l'immagine ritagliata
def save_image(image_path, output_folder, counter, resized_img):
    base_filename = os.path.basename(image_path)
    name, ext = os.path.splitext(base_filename)
    output_path = os.path.join(output_folder, f"{name}_crop{counter}.jpg")
    cv2.imwrite(output_path, resized_img)
    print(f"Cropped image saved to {output_path}")


def detect_and_crop(image_path, output_folder, single_padding):
    global NET, OUTPUT_LAYER, CLASS_INDICES
    range_confidence = 0
    img = cv2.imread(image_path)
    height, width, nonciserve= img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    NET.setInput(blob)
    outs = NET.forward(OUTPUT_LAYER)

    class_ids = []
    confidences = []
    boxes = []

    if single_padding != 0:
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.4 and class_id in CLASS_INDICES.values():
                    #calcola le coordinate del centro dell'oggetto detectato
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    #altezza e larghezza della bounding box
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rendere la bounding box quadrata
                    side = max(w, h) + single_padding
                    x = max(int(center_x - side / 2), 0)
                    y = max(int(center_y - side / 2), 0)
                    side = min(side, width - x, height - y)
                    #i 4 dati che servono poi per poter effettuare il confronto tra le varie bounding box, centro e lati
                    boxes.append([x, y, side, side])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        #se non c'è la confidenza di 0.4 e il grado di sovrapposizione tra le varie bounding box è inferiore a 0.6 scarta quello con confidenza inferiore
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, side, _ = boxes[i]
                cropped_img = img[y:y+side, x:x+side]
                resized_img = cv2.resize(cropped_img, (480, 480))
                save_image(image_path, output_folder, i, resized_img)

    else:
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if "STOP" in output_folder:
                    range_confidence = 0.2
                else:
                    range_confidence = 0.6
                if confidence > range_confidence and class_id in CLASS_INDICES.values():
                    # Salva l'immagine intera
                    save_image(image_path, output_folder, "original", img)

if __name__ == "__main__":
    main()
