import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Carica il modello salvato
model = load_model("C:\\Users\\rocco\\OneDrive\\Desktop\\OUTPUT_Dataset_Image\\final_model_alexnet.h5", compile=True)

class_names = {0: 'SPEED_LIMITER_30', 1: 'SPEED_LIMITER_60', 2: 'SPEED_LIMITER_90', 3: 'STOP_SIGN'}

test_dir = 'C:\\Users\\rocco\\OneDrive\\Desktop\\DMRC-1\\Assets\\RAW_TESTING_Dataset'

batch_size = 32
target_size = (227, 227)
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

def evaluate_model(test_generator, model):
    predictions = model.predict(test_generator, steps=test_generator.samples // batch_size + 1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Etichette vere
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())
    
    cm = confusion_matrix(true_classes, predicted_classes)
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    f1 = f1_score(true_classes, predicted_classes, average='weighted')
    precision = precision_score(true_classes, predicted_classes, average='weighted')
    recall = recall_score(true_classes, predicted_classes, average='weighted')
    
    # Stampa le metriche
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)
    print("\nF1 Score:", f1)
    print("Precision:", precision)
    print("Recall:", recall)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Valutiamo il modello
evaluate_model(test_generator, model)
