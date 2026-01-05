import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from dataset_to_descriptor import dataset_to_descriptor
from descriptor import color_histogram, fusion_histrogram, lbp_histogram, local_desc

image_dir = "data/caltech101_subset/caltech101_subset"
labels_file = "data/caltech101_subset/caltech101_subset.files"


def load_labels():
    """Load labels from the .files file and return a dict mapping filename to label."""
    labels = {}
    with open(labels_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                # Format: caltech101_subset/filename.jpg label
                filepath = parts[0]
                label = parts[1]
                # Extract just the filename
                filename = filepath.split("/")[-1]
                labels[filename] = label
    return labels


def make_dataset():
    """Create dataset with descriptors and their corresponding labels."""
    descriptors = dataset_to_descriptor(image_dir, local_desc)
    labels_map = load_labels()

    X = []
    y = []

    for image_path, desc in descriptors.items():
        if image_path in labels_map:
            X.append(desc)
            y.append(labels_map[image_path])

    return np.array(X), np.array(y)


def train_eval():
    """Train LogisticRegression and evaluate with accuracy and confusion matrix."""
    X, y = make_dataset()

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train LogisticRegression
    clf = LogisticRegression(max_iter=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predict on test set
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Print class labels for reference
    classes = clf.classes_
    print("\nClass labels:", classes)

    return clf, acc, cm


if __name__ == "__main__":
    train_eval()
