from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import os
import matplotlib.pyplot as plt
from skimage.feature import hog
import random
import cv2
import joblib


WIDTH, HEIGHT = 360, 360


def extract_hog_features(image):
    # Calculate HOG features
    hog_features = hog(
        image,
        orientations=8,
        pixels_per_cell=(2, 2),
        cells_per_block=(1, 1),
        visualize=False,
    )
    return hog_features


def get_all_subfolders(directory):
    subfolders = []
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            subfolders.append(os.path.join(root, dir))
    return subfolders


# Define a function to save HOG features and labels
def save_features(features, labels, feature_path, label_path):
    joblib.dump(features, feature_path)
    joblib.dump(labels, label_path)
    print("Features and labels saved.")


# Define a function to load HOG features and labels
def load_features(feature_path, label_path):
    if os.path.exists(feature_path) and os.path.exists(label_path):
        features = joblib.load(feature_path)
        labels = joblib.load(label_path)
        print("Features and labels loaded.")
        return features, labels
    else:
        return None, None


def load_and_extract_features_train_val(directory, feature_path, label_path):
    X, y = load_features(feature_path, label_path)
    if X is not None and y is not None:
        return X, y

    print(
        "\t====================   FEATURE'S EXTRACTION PHASE [TRAIN-VALIDATION]  ===================="
    )
    X = []
    y = []
    list = get_all_subfolders(directory)
    list = [item for item in list if "Padding" in item]
    total_items = len(list)
    current = 1
    for dir in list:
        label = ""
        if "SPEED_LIMITER_30" in dir:
            label = "SPEED_LIMITER_30"
        elif "SPEED_LIMITER_60" in dir:
            label = "SPEED_LIMITER_60"
        elif "SPEED_LIMITER_90" in dir:
            label = "SPEED_LIMITER_90"
        elif "STOP" in dir:
            label = "STOP_SIGN"

        for filename in os.listdir(dir):
            image_path = os.path.join(dir, filename)
            # Load image using OpenCV
            img = cv2.imread(image_path)

            # Resize image to (480, 480)
            img_resized = cv2.resize(img, (HEIGHT, WIDTH))

            # Convert image to grayscale
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

            # Calculate HOG features
            hog_features = extract_hog_features(img_gray)

            X.append(hog_features)
            y.append(label)
        print(f"{label}: {current}/{total_items} DONE")
        current += 1

    save_features(X, y, feature_path, label_path)
    return X, y


def load_and_extract_features_test(directory, feature_path, label_path):
    X, y = load_features(feature_path, label_path)
    if X is not None and y is not None:
        return X, y

    print(
        "\t====================   FEATURE'S EXTRACTION PHASE [TESTING]   ===================="
    )
    X = []
    y = []
    list = get_all_subfolders(directory)
    total_items = len(list)
    current = 1
    for dir in list:
        label = ""
        if "SPEED_LIMITER_30" in dir:
            label = "SPEED_LIMITER_30"
        elif "SPEED_LIMITER_60" in dir:
            label = "SPEED_LIMITER_60"
        elif "SPEED_LIMITER_90" in dir:
            label = "SPEED_LIMITER_90"
        elif "STOP" in dir:
            label = "STOP_SIGN"

        for filename in os.listdir(dir):
            image_path = os.path.join(dir, filename)
            # Load image using OpenCV
            img = cv2.imread(image_path)
            # Resize image to (480, 480)
            img_resized = cv2.resize(img, (HEIGHT, WIDTH))
            # Convert image to grayscale
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            # Calculate HOG features
            hog_features = extract_hog_features(img_gray)
            X.append(hog_features)
            y.append(label)
        print(f"{label}: {current}/{total_items} DONE")
        current += 1

    save_features(X, y, feature_path, label_path)
    return X, y


# Define a function to train a Random Forest classifier
def train_random_forest(X_train, y_train):
    print(
        "\t==============================   TRAINING PHASE   =============================="
    )

    rf_classifier = RandomForestClassifier(
        n_estimators=1024,
        criterion="gini",
        n_jobs=-1,
        max_depth=128,
        oob_score=True,
    )
    rf_classifier.fit(X_train, y_train)

    joblib.dump(rf_classifier, f"./RF_Model_{WIDTH}.pkl")
    print("SAVED RF MODEL...")

    return rf_classifier


def main():
    print("Executing Main...")
    train_feature_path = f"./train_features_{WIDTH}.pkl"
    train_label_path = f"./train_labels_{WIDTH}.pkl"
    test_feature_path = f"./test_features_{WIDTH}.pkl"
    test_label_path = f"./test_labels_{WIDTH}.pkl"

    # Load and extract features from training data
    train_X, train_y = load_and_extract_features_train_val(
        "../../Assets/CROPPED_Dataset", train_feature_path, train_label_path
    )

    # Train Random Forest classifiers
    rf_classifier = train_random_forest(train_X, train_y)
    # rf_classifier = joblib.load(f"./RF_Model_{WIDTH}.pkl")

    # Load and extract features from testing data
    test_X, test_y = load_and_extract_features_test(
        "../../Assets/RAW_TESTING_Dataset", test_feature_path, test_label_path
    )

    predictions = rf_classifier.predict(test_X)
    score = rf_classifier.score(test_X, test_y)

    labels = ["SPEED_LIMITER_30", "SPEED_LIMITER_60", "SPEED_LIMITER_90", "STOP_SIGN"]

    print(f"Model OOB_Score: {rf_classifier.oob_score_}")
    print(f"Model Score: {score}")

    cm = confusion_matrix(predictions, test_y)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)
    plt.savefig(f"./ConfusionMatrix_{WIDTH}.png")


if __name__ == "__main__":
    main()
