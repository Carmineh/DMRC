import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# Carica il modello salvato
#model = load_model('/content/drive/MyDrive/modelli-se4ai/model-092-5layer.keras', compile=True)
#model = load_model('/content/drive/MyDrive/modelli-se4ai/best_model-081-4layer.keras', compile=True)
# model = load_model('/content/drive/MyDrive/modelli-se4ai/my_model.keras', compile=True)
#model = load_model('/content/drive/MyDrive/modelli-se4ai/best_model.keras', compile=True)
#model = load_model('/content/drive/MyDrive/modelli-se4ai/model-081-4layer.keras', compile=True)
#model = load_model('/content/drive/MyDrive/modelli-se4ai/model.keras', compile=True)
#model = load_model('/content/drive/MyDrive/modelli-se4ai/model-profondo.keras', compile=True)
model = load_model("C:\\Users\\rocco\\OneDrive\\Desktop\\OUTPUT_Dataset_Nonoriginal\\model.keras", compile=True)
# Funzione per calcolare le metriche
def evaluate_model(test_tensors, test_labels, model):
    # Previsioni
    predictions = model.predict(test_tensors)
    predicted_classes = np.argmax(predictions, axis=1)

    # Confusion matrix
    cm = confusion_matrix(test_labels, predicted_classes)
    
    # Classification report
    report = classification_report(test_labels, predicted_classes, target_names=list(class_names.values()))
    
    # F1 Score, Precision, Recall
    f1 = f1_score(test_labels, predicted_classes, average='weighted')
    precision = precision_score(test_labels, predicted_classes, average='weighted')
    recall = recall_score(test_labels, predicted_classes, average='weighted')
    
    # Stampa le metriche
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)
    print("\nF1 Score:", f1)
    print("Precision:", precision)
    print("Recall:", recall)
    
    # Visualizza la confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(class_names.values()), yticklabels=list(class_names.values()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Mappa delle classi
class_names = {0: 'SPEED_LIMITER_30', 1: 'SPEED_LIMITER_60', 2: 'SPEED_LIMITER_90', 3: 'STOP_SIGN'}

# Carica i tensori di test e le etichette
output_dir = 'C:\\Users\\rocco\\OneDrive\\Desktop\\OUTPUT_Dataset'
test_tensors = np.load(os.path.join(output_dir, 'test_tensors.npy'))
tensors = tf.image.resize(test_tensors, (224,224))
test_labels = np.load(os.path.join(output_dir, 'test_labels.npy'))

# Valuta il modello
evaluate_model(tensors, test_labels, model)
