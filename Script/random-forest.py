import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib  # Per salvare il modello

# Definisci un generatore per caricare i dati in modo incrementale
def data_generator(tensors_file, labels_file, batch_size=32):
    tensors = np.load(tensors_file, mmap_mode='r')  # Utilizza mmap_mode='r' per la lettura incrementale
    labels = np.load(labels_file)
    num_samples = len(tensors)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        yield tensors[batch_indices], labels[batch_indices]

# Trasforma le etichette in formato one-hot durante il preprocessamento
def preprocess(tensors, labels, num_classes):
    labels = tf.one_hot(labels, depth=num_classes)
    return tensors, labels

def main():
    output_dir = 'C:\\Users\\rocco\\OneDrive\\Desktop\\DMRC-1\\OUTPUT_Dataset'
    save_dir = 'C:\\Users\\rocco\\OneDrive\\Desktop\\DMRC-1\\trained-model'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Creazione del dataset TensorFlow dai file `.npy` utilizzando il generatore
    train_tensors_file = os.path.join(output_dir, 'train_tensors.npy')
    train_labels_file = os.path.join(output_dir, 'train_labels.npy')
    val_tensors_file = os.path.join(output_dir, 'val_tensors.npy')
    val_labels_file = os.path.join(output_dir, 'val_labels.npy')

    # Carica i dati di addestramento e di validazione
    train_tensors = np.load(train_tensors_file)
    train_labels = np.load(train_labels_file)
    val_tensors = np.load(val_tensors_file)
    val_labels = np.load(val_labels_file)

    # Ridimensiona i tensori per adattarli al modello Random Forest
    train_tensors = train_tensors.reshape(train_tensors.shape[0], -1)
    val_tensors = val_tensors.reshape(val_tensors.shape[0], -1)

    # Creazione del modello Random Forest
    model = RandomForestClassifier(n_estimators=128, criterion='entropy', max_depth=20, n_jobs=-1, oob_score=True)
    model.fit(train_tensors, train_labels)

    # Predizione
    y_pred = model.predict(val_tensors)

    # Stampa del risultato
    print("Accuracy:", accuracy_score(val_labels, y_pred))
    print("Classification Report:\n", classification_report(val_labels, y_pred))

    # Etichette dei segnali stradali
    label_names = ['SPEED_LIMITER_30', 'SPEED_LIMITER_60', 'SPEED_LIMITER_90', 'STOP_SIGN']

    # Confusion Matrix
    cm = confusion_matrix(val_labels, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)

    # Plot della Confusion Matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))  # Salva l'immagine
    plt.show()
 
    # Errore Out-Of-Bag
    oob_error = 1 - model.oob_score_
    print("OOB Error:", oob_error)
    print("OOB Score:", model.oob_score_)

    # Salva il modello
    model_file = os.path.join(save_dir, 'random_forest_model.pkl')
    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}")

if __name__ == "__main__":
    main()
