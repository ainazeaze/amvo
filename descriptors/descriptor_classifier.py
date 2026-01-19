"""Image classification using descriptors and logistic regression."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from dataset_to_descriptor import dataset_to_descriptor
from descriptor import color_histogram, fusion_histogram, lbp_histogram, local_desc

IMAGE_DIR = "data/caltech101_subset/caltech101_subset"
LABELS_FILE = "data/caltech101_subset/caltech101_subset.files"


def load_labels(labels_file=LABELS_FILE):
    """Load labels from a .files annotation file.

    Args:
        labels_file: Path to the file containing image paths and labels.
            Expected format: "path/to/image.jpg label" per line.

    Returns:
        Dictionary mapping image filenames to their class labels.
    """
    labels = {}

    with open(labels_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                filepath = parts[0]
                label = parts[1]
                filename = filepath.split("/")[-1]
                labels[filename] = label

    return labels


def make_dataset(image_dir=IMAGE_DIR, descriptor_func=local_desc):
    """Create a dataset of descriptors with corresponding labels.

    Args:
        image_dir: Path to the directory containing images.
        descriptor_func: Function to compute image descriptors.

    Returns:
        Tuple of (X, y) where X is a numpy array of descriptors
        and y is a numpy array of labels.
    """
    descriptors = dataset_to_descriptor(image_dir, descriptor_func)
    labels_map = load_labels()

    X = []
    y = []

    for image_path, desc in descriptors.items():
        if image_path in labels_map:
            X.append(desc)
            y.append(labels_map[image_path])

    return np.array(X), np.array(y)


def train_eval(descriptor_func=local_desc, test_size=0.2, random_state=42):
    """Train a logistic regression classifier and evaluate its performance.

    Args:
        descriptor_func: Function to compute image descriptors.
        test_size: Fraction of data to use for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (classifier, accuracy, confusion_matrix).
    """
    X, y = make_dataset(descriptor_func=descriptor_func)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClass labels:", clf.classes_)

    return clf, acc, cm


if __name__ == "__main__":
    train_eval()
