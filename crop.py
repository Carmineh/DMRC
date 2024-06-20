import shutil
import cv2
import numpy as np
import os

weights_path = 'C:/Users/rocco/OneDrive/Desktop/CARLA-weight/CARLA-weight/yolov3-tiny-obj_5000.weights'
config_path = 'C:/Users/rocco/OneDrive/Desktop/CARLA-weight/CARLA-weight/yolov3-tiny-obj.cfg'
labels_path = 'C:/Users/rocco/OneDrive/Desktop/CARLA-weight/classes.names'

# YOLO
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO labels
with open(labels_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

class_indices = {
    "STOP": 11,  
    "30": 0,    
    "60": 1,    
    "90": 2     
}


input_folder = 'Assets/RAW_Dataset/STOP_SIGN/clear_night'
output_folder = 'cropped/STOP_SIGN/clear_night'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
else:
    shutil.rmtree(output_folder)
    os.makedirs(output_folder)

padding = 60  # Più lo aumentiamo e più la ritaglia larga

def detect_and_crop(image_path, output_folder):
    img = cv2.imread(image_path)
    height, width, channels = img.shape

 
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # Se la confidenza non è almeno 0.01 scarta
            if confidence > 0.01 and class_id in class_indices.values():
                # Oggetto rilevato
                center_x = int(detection[0] * width) #queste due sono i centri dell'oggetto rilevato
                center_y = int(detection[1] * height)
                w = int(detection[2] * width) # queste due sono larghezza e altezza dell'oggetto rilevato
                h = int(detection[3] * height)
                # x = max(int(center_x - w / 2 - padding), 0) #coordinate vertice superiore del detection, con padding
                # y = max(int(center_y - h / 2 - padding), 0)
                # w = min(int(w + 2 * padding), width - x) #trova larghezza e l'altezza della bounding box
                # h = min(int(h + 2 * padding), height - y)
                # boxes.append([x, y, w, h])
                
                side = max(w, h) + 2 * padding # rendiamo la bounding box quadrata per eveitare distorsioni
                x = max(int(center_x - side / 2 ), 0) #coordinate vertice superiore del detection, con padding
                y = max(int(center_y - side / 2 ), 0)
                side = min(side, width - x, height - y) #trova larghezza e l'altezza della bounding box. In questo caso abbiamo solo side perchè deve essere quadrata
                boxes.append([x, y, side, side])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # applica la soppressione non massima (NMS) per eliminare le rilevazioni sovrapposte
    #Questo fa si che alla fine mi restituisce una sola bounding box, con confidenza maggiore e 0.01 e non sovrapposte
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.01, 1)

    # Crop and save the detected road sign images
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            cropped_img = img[y:y+h, x:x+w]
            resized_img = cv2.resize(cropped_img, (480, 480))
            base_filename = os.path.basename(image_path)
            name, ext = os.path.splitext(base_filename)
            output_path = os.path.join(output_folder, f"{name}_crop{i}.jpg")
            cv2.imwrite(output_path, resized_img)
            print(f"Cropped image saved to {output_path}")

# Iterate over all images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        image_path = os.path.join(input_folder, filename)
        detect_and_crop(image_path, output_folder)
