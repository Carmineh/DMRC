import shutil
import cv2
import numpy as np
import os

def main():
    padding = [0, 20, 40, 60, 80]
    
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

def save_image(image_path, output_folder, counter, resized_img):
    base_filename = os.path.basename(image_path)
    name, ext = os.path.splitext(base_filename)
    output_path = os.path.join(output_folder, f"{name}_crop{counter}.jpg")
    cv2.imwrite(output_path, resized_img)
    print(f"Cropped image saved to {output_path}")

def detect_and_crop(image_path, output_folder, single_padding):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 20 < w < 200 and 20 < h < 200:  # dimensioni indicative dei cartelli stradali
            side = max(w, h) + single_padding
            x = max(int(x - (side - w) / 2), 0)
            y = max(int(y - (side - h) / 2), 0)
            side = min(side, img.shape[1] - x, img.shape[0] - y)
            cropped_img = img[y:y+side, x:x+side]
            resized_img = cv2.resize(cropped_img, (480, 480))
            
            # Verifica delle texture (presenza del 30, 60, 90 o STOP)
            if contains_sign(resized_img, gray[y:y+side, x:x+side]):
                save_image(image_path, output_folder, f"{x}_{y}", resized_img)

def contains_sign(img, gray_img):
    template_30 = cv2.imread('templates/30.jpg', 0)
    template_60 = cv2.imread('templates/60.jpg', 0)
    template_90 = cv2.imread('templates/90.jpg', 0)
    template_stop = cv2.imread('templates/stop.jpg', 0)
    templates = [template_30, template_60, template_90, template_stop]
    
    for template in templates:
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.6
        loc = np.where(res >= threshold)
        if len(loc[0]) > 0:
            return True
    return False

if __name__ == "__main__":
    main()
