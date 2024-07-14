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

# Mappa delle classi
class_names = {0: 'SPEED_LIMITER_30', 1: 'SPEED_LIMITER_60', 2: 'SPEED_LIMITER_90', 3: 'STOP_SIGN'}

# Directory dei dati di test
test_dir = 'C:\\Users\\rocco\\OneDrive\\Desktop\\DMRC-1\\Assets\\RAW_TESTING_Dataset'

# Parametri del generatore di dati
batch_size = 32
target_size = (227, 227)

# Generazione dei dati di test
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

def evaluate_model(test_generator, model):
    # Effettua le previsioni per un batch
    x_valid, label_batch = next(iter(test_generator))
    predictions = model.predict(x_valid)
    predicted_classes = np.argmax(predictions, axis=1)

    # Set up the figure
    fig = plt.figure(figsize=(10, 6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    # Plot the images
    for i in range(len(x_valid)):
        ax = fig.add_subplot(2, 4, i + 1, xticks=[], yticks=[])
        ax.imshow(x_valid[i,:], cmap=plt.cm.gray_r, interpolation='nearest')
        true_label = np.argmax(label_batch[i])
        if predicted_classes[i] == true_label:
            ax.text(3, 17, class_names[predicted_classes[i]], color='blue', fontsize=14)
        else:
            ax.text(3, 17, class_names[predicted_classes[i]], color='red', fontsize=14)
    plt.show()

    # Etichette vere
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)

    # Classification report
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)

    # F1 Score, Precision, Recall
    f1 = f1_score(true_classes, predicted_classes, average='weighted')
    precision = precision_score(true_classes, predicted_classes, average='weighted')
    recall = recall_score(true_classes, predicted_classes, average='weighted')

    # Visualizza la confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Stampa le metriche
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)
    print("\nF1 Score:", f1)
    print("Precision:", precision)
    print("Recall:", recall)

# Valuta il modello
evaluate_model(test_generator, model)
