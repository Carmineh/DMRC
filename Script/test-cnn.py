import tensorflow as tf
import numpy as np
import cv2
from keras.models import load_model

# Percorso del modello salvato
model_path = 'C:\\Users\\rocco\\OneDrive\\Desktop\\DMRC-1\\OUTPUT_Dataset\\model.h5'

# Carica il modello salvato
model = load_model('C:\\Users\\rocco\\OneDrive\\Desktop\\DMRC-1\\OUTPUT_Dataset\\my_model.keras')

# Funzione per preprocessare l'immagine
def preprocess_image(image_path):
    # Carica l'immagine
    image = cv2.imread(image_path)
    # Ridimensiona l'immagine alle dimensioni richieste dal modello (480x480)
    image = cv2.resize(image, (480, 480))
    # Converti l'immagine in un array numpy
    image = np.array(image)
    # Normalizza i valori dei pixel tra 0 e 1
    image = image / 255.0
    # Aggiungi una dimensione batch (1, 480, 480, 3)
    image = np.expand_dims(image, axis=0)
    return image

# Percorso dell'immagine da prevedere
image_path = 'C:\\Users\\rocco\\OneDrive\\Desktop\\DMRC-1\\TEST_FIle\\test_image_stop.png'
#image_path = 'C:\\Users\\rocco\\OneDrive\\Desktop\\DMRC-1\\Assets\\RAW_Dataset\\STOP_SIGN\\stop_clearnoon\\STOP_CLEARNOON_23.png'

# Preprocessa l'immagine
preprocessed_image = preprocess_image(image_path)

# Stampa l'immagine preprocessata per la verifica
print(f"Preprocessed image shape: {preprocessed_image.shape}")
print(preprocessed_image)

# Fai una previsione
predictions = model.predict(preprocessed_image)

# Stampa le previsioni grezze
print(f"Raw predictions: {predictions}")

# Mappa delle classi
class_names = {0: 'SPEED_LIMITER_30', 1: 'SPEED_LIMITER_60', 2: 'SPEED_LIMITER_90', 3: 'STOP_SIGN'}

# Ottieni la classe prevista (la classe con la probabilità più alta)
predicted_class = np.argmax(predictions)
predicted_class_name = class_names[predicted_class]

# Stampa la previsione
print(f"Predicted class: {predicted_class} ({predicted_class_name})")
