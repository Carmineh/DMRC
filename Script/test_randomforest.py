import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog, local_binary_pattern
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def apply_canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blurred, 50, 150)
    return canny

def extract_features(image):
    canny_image = apply_canny(image)
    fd_hog, _ = hog(canny_image, orientations=12, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True, feature_vector=True)
    lbp = local_binary_pattern(canny_image, P=24, R=3, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 26), range=(0, 24), density=True)
    features = np.concatenate((fd_hog, lbp_hist))
    return features

model_path = "C:\\Users\\rocco\\OneDrive\\Desktop\\OUTPUT_Dataset_RandomForest\\random_forest_model.pkl"
pca_path = "C:\\Users\\rocco\\OneDrive\\Desktop\\OUTPUT_Dataset_RandomForest\\pca_model.pkl"
rf = joblib.load(model_path)
pca = joblib.load(pca_path)

test_dir = 'C:\\Users\\rocco\\OneDrive\\Desktop\\DMRC-1\\Assets\\RAW_TESTING_Dataset'

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(480, 480), #Cambiare qua se hai cambiato nel training la shape
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

steps_per_epoch = test_generator.samples

X_test, y_test = [], []
for _ in range(steps_per_epoch):
    batch_x, batch_y = next(test_generator)
    for x, y in zip(batch_x, batch_y):
        features = extract_features((x * 255).astype(np.uint8))
        X_test.append(features)
        y_test.append(np.argmax(y))

X_test = np.array(X_test)
y_test = np.array(y_test)

X_test_pca = pca.transform(X_test)

y_pred = rf.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=list(test_generator.class_indices.keys()))

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(test_generator.class_indices.keys()), yticklabels=list(test_generator.class_indices.keys()))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", report)
