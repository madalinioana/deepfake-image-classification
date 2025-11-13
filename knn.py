import os
import csv
import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data_path = "/kaggle/input/deepfake-classification"
train_csv = os.path.join(data_path, "train.csv")
val_csv = os.path.join(data_path, "validation.csv")
train_images_path = os.path.join(data_path, "train")
val_images_path = os.path.join(data_path, "validation")

def load_labels(csv_path):
    images, labels = [], []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for img_id, lbl in reader:
            images.append(f"{img_id}.png")
            labels.append(int(lbl))
    return images, labels

train_images, train_labels = load_labels(train_csv)
val_images, val_labels = load_labels(val_csv)

def extract_features(images, image_path, size=(100, 100)):
    x = []
    for image in images:
        img = Image.open(os.path.join(image_path, image)).convert("RGB")
        img = img.resize(size)
        arr = np.array(img, dtype=np.float32) / 255.0
        x.append(arr.flatten())
    return np.stack(x)

x_train = extract_features(train_images, train_images_path)
y_train = np.array(train_labels)
x_val = extract_features(val_images, val_images_path)
y_val = np.array(val_labels)

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(x_train, y_train)

y_pred = knn.predict(x_val)
acc = accuracy_score(y_val, y_pred)

print(f"Accuracy: {acc:.4f}")
