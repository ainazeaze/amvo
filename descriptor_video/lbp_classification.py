"""
Question 7.6: Classification d'actions avec les descripteurs LBP et LBP-TOP.

Ce module reproduit les expérimentations de classification (SVM + leave-one-out)
en utilisant les descripteurs LBP et LBP-TOP au lieu de HOG/HOF.
"""

import os

import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from descriptor_video.lbp_bovw import compute_bovw_dataset


def classify_leave_one_out(X, y, kernel="rbf", C=1.0):
    """Perform leave-one-out classification using SVM.

    Args:
        X: numpy array of shape (n_samples, n_features) - feature vectors (BoVW)
        y: numpy array of shape (n_samples,) - labels
        kernel: SVM kernel type ("rbf", "linear", "poly", etc.)
        C: SVM regularization parameter

    Returns:
        Tuple of:
            - accuracy: mean classification accuracy
            - predictions: predicted labels for each sample
            - true_labels: true labels (same as y)
    """
    # Encode string labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    loo = LeaveOneOut()
    predictions = np.zeros(len(y_encoded), dtype=int)

    n_samples = len(y_encoded)
    for i, (train_idx, test_idx) in enumerate(loo.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        # Train SVM
        clf = SVC(kernel=kernel, C=C)
        clf.fit(X_train, y_train)

        # Predict
        predictions[test_idx] = clf.predict(X_test)

        if (i + 1) % 30 == 0:
            print(f"Leave-one-out: {i + 1}/{n_samples} done...")

    # Calculate accuracy
    accuracy = np.mean(predictions == y_encoded)

    # Decode predictions back to original labels
    predictions_decoded = label_encoder.inverse_transform(predictions)

    print(f"Leave-one-out completed: {n_samples}/{n_samples}")

    return accuracy, predictions_decoded, y


def run_lbp_classification(
    dataset_file,
    vocabulary_path,
    video_dir="data/videos",
    keypoints_dir="data/keypoints",
    neighborhood_size=3,
    descriptor_type="lbp_top",
    kernel="rbf",
    C=1.0,
):
    """Run a complete classification experiment with LBP/LBP-TOP.

    Args:
        dataset_file: Path to .files dataset file
        vocabulary_path: Path to visual vocabulary .npy file
        video_dir: Directory containing video files
        keypoints_dir: Directory containing .key files
        neighborhood_size: Spatio-temporal neighborhood size
        descriptor_type: "lbp" or "lbp_top"
        kernel: SVM kernel type
        C: SVM regularization parameter

    Returns:
        Tuple of (accuracy, predictions, true_labels, X, y)
    """
    print(f"\n{'=' * 60}")
    print(f"Classification experiment: {descriptor_type.upper()}")
    print(f"{'=' * 60}")

    # Load vocabulary
    print(f"Loading vocabulary from {vocabulary_path}...")
    vocabulary = np.load(vocabulary_path)
    print(
        f"Vocabulary size: {vocabulary.shape[0]}, descriptor dim: {vocabulary.shape[1]}"
    )

    # Compute BoVW vectors for all videos
    print(f"\nComputing BoVW vectors ({descriptor_type})...")
    X, y, video_names = compute_bovw_dataset(
        dataset_file,
        vocabulary,
        video_dir=video_dir,
        keypoints_dir=keypoints_dir,
        neighborhood_size=neighborhood_size,
        descriptor_type=descriptor_type,
    )
    print(f"Dataset: {X.shape[0]} videos, {X.shape[1]} features")

    # Get unique classes
    unique_classes = np.unique(y)
    print(f"Classes ({len(unique_classes)}): {list(unique_classes)}")

    # Run leave-one-out classification
    print(f"\nRunning leave-one-out SVM classification (kernel={kernel}, C={C})...")
    accuracy, predictions, true_labels = classify_leave_one_out(X, y, kernel, C)

    print(f"\n{'=' * 60}")
    print(
        f"ACCURACY ({descriptor_type.upper()}): {accuracy:.4f} ({accuracy * 100:.2f}%)"
    )
    print(f"{'=' * 60}")

    return accuracy, predictions, true_labels, X, y


def compare_lbp_descriptors(
    dataset_file="data/ucf-sports.files",
    vocab_dir="visual_vocabularies",
    video_dir="data/videos",
    keypoints_dir="data/keypoints",
    neighborhood_size=3,
    kernel="rbf",
    C=1.0,
):
    """Compare classification performance of LBP and LBP-TOP descriptors.

    Args:
        dataset_file: Path to .files dataset file
        vocab_dir: Directory containing vocabulary files
        video_dir: Directory containing video files
        keypoints_dir: Directory containing .key files
        neighborhood_size: Spatio-temporal neighborhood size
        kernel: SVM kernel type
        C: SVM regularization parameter

    Returns:
        Dictionary with results for each descriptor type
    """
    results = {}

    descriptor_configs = [
        ("lbp", f"{vocab_dir}/voc_lbp_500.npy"),
        ("lbp_top", f"{vocab_dir}/voc_lbp_top_500.npy"),
    ]

    for desc_type, vocab_path in descriptor_configs:
        if not os.path.exists(vocab_path):
            print(f"\nVocabulary not found: {vocab_path}")
            print("Run lbp_vocabulary.py first to create the vocabulary.")
            continue

        accuracy, predictions, true_labels, X, y = run_lbp_classification(
            dataset_file=dataset_file,
            vocabulary_path=vocab_path,
            video_dir=video_dir,
            keypoints_dir=keypoints_dir,
            neighborhood_size=neighborhood_size,
            descriptor_type=desc_type,
            kernel=kernel,
            C=C,
        )
        results[desc_type] = {
            "accuracy": accuracy,
            "predictions": predictions,
            "true_labels": true_labels,
            "X": X,
            "y": y,
        }

    # Summary
    if results:
        print(f"\n{'=' * 60}")
        print("SUMMARY - Comparison of LBP descriptors")
        print(f"{'=' * 60}")
        for desc_type, res in results.items():
            print(
                f"{desc_type.upper():10s}: {res['accuracy']:.4f} ({res['accuracy'] * 100:.2f}%)"
            )

    return results


def compare_all_descriptors(
    dataset_file="data/ucf-sports.files",
    vocab_dir="visual_vocabularies",
    video_dir="data/videos",
    keypoints_dir="data/keypoints",
    neighborhood_size=3,
    kernel="rbf",
    C=1.0,
):
    """Compare all descriptors: HOG, HOF, HOG+HOF, LBP, LBP-TOP.

    Args:
        Same as compare_lbp_descriptors

    Returns:
        Dictionary with results for all descriptor types
    """
    from descriptor_video.classification import run_classification_experiment

    results = {}

    # HOG/HOF descriptors
    hoghof_configs = [
        ("hoghof", f"{vocab_dir}/voc_hoghof_500.npy", "hoghof"),
        ("hog", f"{vocab_dir}/voc_hog_500.npy", "hog"),
        ("hof", f"{vocab_dir}/voc_hof_500.npy", "hof"),
    ]

    for name, vocab_path, desc_type in hoghof_configs:
        if os.path.exists(vocab_path):
            accuracy, predictions, true_labels, X, y = run_classification_experiment(
                dataset_file=dataset_file,
                vocabulary_path=vocab_path,
                keypoints_dir=keypoints_dir,
                descriptor_type=desc_type,
                kernel=kernel,
                C=C,
            )
            results[name] = {"accuracy": accuracy}

    # LBP descriptors
    lbp_configs = [
        ("lbp", f"{vocab_dir}/voc_lbp_500.npy"),
        ("lbp_top", f"{vocab_dir}/voc_lbp_top_500.npy"),
    ]

    for desc_type, vocab_path in lbp_configs:
        if os.path.exists(vocab_path):
            accuracy, predictions, true_labels, X, y = run_lbp_classification(
                dataset_file=dataset_file,
                vocabulary_path=vocab_path,
                video_dir=video_dir,
                keypoints_dir=keypoints_dir,
                neighborhood_size=neighborhood_size,
                descriptor_type=desc_type,
                kernel=kernel,
                C=C,
            )
            results[desc_type] = {"accuracy": accuracy}

    # Final summary
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY - All descriptors comparison")
    print(f"{'=' * 60}")
    print(f"{'Descriptor':<12} {'Accuracy':<12}")
    print("-" * 24)
    for desc_type in ["hoghof", "hog", "hof", "lbp", "lbp_top"]:
        if desc_type in results:
            acc = results[desc_type]["accuracy"]
            print(f"{desc_type.upper():<12} {acc * 100:>6.2f}%")
        else:
            print(f"{desc_type.upper():<12} {'N/A':>6}")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Question 7.6: Classification avec LBP et LBP-TOP")
    print("=" * 60)

    # Compare LBP and LBP-TOP
    results = compare_lbp_descriptors(
        dataset_file="data/ucf-sports.files",
        vocab_dir="visual_vocabularies",
        video_dir="data/videos",
        keypoints_dir="data/keypoints",
        neighborhood_size=3,
        kernel="rbf",
        C=1.0,
    )
