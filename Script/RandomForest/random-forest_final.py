import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tqdm import tqdm


def apply_canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blurred, 50, 150)
    return canny


def extract_features(image):
    canny_image = apply_canny(image)
    fd_hog, _ = hog(
        canny_image,
        orientations=12,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        feature_vector=True,
    )
    lbp = local_binary_pattern(canny_image, P=24, R=3, method="uniform")
    lbp_hist, _ = np.histogram(
        lbp.ravel(), bins=np.arange(0, 26), range=(0, 24), density=True
    )
    features = np.concatenate((fd_hog, lbp_hist))
    return features


def get_class_weights(directory):
    classes = os.listdir(directory)
    num_samples_per_class = {}
    for cls in classes:
        path = os.path.join(directory, cls)
        count = 0
        for root, dirs, files in os.walk(path):
            count += len(
                [
                    file
                    for file in files
                    if file.endswith(".jpg") or file.endswith(".png")
                ]
            )
        num_samples_per_class[cls] = count
    total_samples = sum(num_samples_per_class.values())
    class_weights = {
        i: total_samples / (len(classes) * num_samples_per_class[cls])
        for i, cls in enumerate(classes)
    }
    return class_weights


base_dir = "../..//Assets/CROPPED_Dataset"
output_dir = "./OUTPUT_Dataset_RandomForest"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

datagen = ImageDataGenerator(rescale=1.0 / 255)
generator = datagen.flow_from_directory(
    base_dir,
    target_size=(480, 480),
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
)
steps_per_epoch = generator.samples // generator.batch_size

X, y = [], []
for batch_x, batch_y in tqdm(generator, total=steps_per_epoch):
    for x, label in zip(batch_x, batch_y):
        features = extract_features((x * 255).astype(np.uint8))
        X.append(features)
        y.append(np.argmax(label))
    if len(X) >= generator.samples:
        break

X = np.array(X)
y = np.array(y)

initial_pca = PCA().fit(X)
cumulative_variance = np.cumsum(initial_pca.explained_variance_ratio_)
n_components = np.argmax(cumulative_variance >= 0.95) + 1
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

# Bilanciamento delle classi
class_weights = get_class_weights(base_dir)

params = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [None, 20, 30, 40, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False],
}

rf = RandomForestClassifier(random_state=42, class_weight=class_weights)
clf = GridSearchCV(rf, params, cv=3, n_jobs=-1, verbose=2)
clf.fit(X_pca, y)

print(f"Best parameters: {clf.best_params_}")
# Salvataggio del modello
joblib.dump(clf.best_estimator_, os.path.join(output_dir, "random_forest_model.pkl"))
joblib.dump(pca, os.path.join(output_dir, "pca_model.pkl"))

# Risultati dell'addestramento
accuracy = accuracy_score(y, clf.predict(X))
print(f"Accuracy: {accuracy * 100:.2f}%")

plt.figure(figsize=(8, 4))
plt.bar(["Training Accuracy"], [accuracy])
plt.ylabel("Accuracy")
plt.title("Training Accuracy Overview")
plt.savefig(os.path.join(output_dir, "training_accuracy_plot.png"))
plt.show()
