"""
Classification d'actions par sacs de mots visuels spatio-temporels.

Ce module implémente le pipeline de classification utilisant:
- Descripteurs locaux HOG, HOF ou HOG+HOF
- Bag of Visual Words (BoVW)
- SVM multiclasse
- Validation leave-one-out
"""

import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from descriptor_video.key_desc import compute_bovw_dataset


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


def run_classification_experiment(
    dataset_file,
    vocabulary_path,
    keypoints_dir="data/keypoints",
    descriptor_type="hoghof",
    kernel="rbf",
    C=1.0,
):
    """Run a complete classification experiment.

    Args:
        dataset_file: Path to .files dataset file
        vocabulary_path: Path to visual vocabulary .npy file
        keypoints_dir: Directory containing .key files
        descriptor_type: "hoghof", "hog", or "hof"
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
        dataset_file, vocabulary, keypoints_dir, descriptor_type
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


def compare_descriptors(
    dataset_file,
    vocab_dir="visual_vocabularies",
    keypoints_dir="data/keypoints",
    kernel="rbf",
    C=1.0,
):
    """Compare classification performance of HOG, HOF, and HOG+HOF descriptors.

    Args:
        dataset_file: Path to .files dataset file
        vocab_dir: Directory containing vocabulary files
        keypoints_dir: Directory containing .key files
        kernel: SVM kernel type
        C: SVM regularization parameter

    Returns:
        Dictionary with results for each descriptor type
    """
    results = {}

    descriptor_configs = [
        ("hoghof", f"{vocab_dir}/voc_hoghof_500.npy"),
        ("hog", f"{vocab_dir}/voc_hog_500.npy"),
        ("hof", f"{vocab_dir}/voc_hof_500.npy"),
    ]

    for desc_type, vocab_path in descriptor_configs:
        accuracy, predictions, true_labels, X, y = run_classification_experiment(
            dataset_file=dataset_file,
            vocabulary_path=vocab_path,
            keypoints_dir=keypoints_dir,
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
    print(f"\n{'=' * 60}")
    print("SUMMARY - Comparison of descriptors")
    print(f"{'=' * 60}")
    for desc_type, res in results.items():
        print(
            f"{desc_type.upper():10s}: {res['accuracy']:.4f} ({res['accuracy'] * 100:.2f}%)"
        )

    return results


def classify_leave_one_out_augmented(
    X, y, video_names, X_aug, y_aug, video_names_aug, kernel="rbf", C=1.0
):
    """Perform leave-one-out classification with data augmentation.

    For each test sample, the training set includes:
    - All other original samples
    - All augmented samples EXCEPT the augmented version of the test sample

    Args:
        X: numpy array of shape (n_samples, n_features) - original feature vectors
        y: numpy array of shape (n_samples,) - original labels
        video_names: numpy array of video names (original)
        X_aug: numpy array of shape (n_samples, n_features) - augmented feature vectors
        y_aug: numpy array of shape (n_samples,) - augmented labels
        video_names_aug: numpy array of augmented video names
        kernel: SVM kernel type
        C: SVM regularization parameter

    Returns:
        Tuple of (accuracy, predictions, true_labels)
    """
    # Encode string labels to integers
    label_encoder = LabelEncoder()
    # Fit on all labels (original + augmented)
    all_labels = np.concatenate([y, y_aug])
    label_encoder.fit(all_labels)

    y_encoded = label_encoder.transform(y)
    y_aug_encoded = label_encoder.transform(y_aug)

    n_samples = len(y_encoded)
    predictions = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        # Test sample
        X_test = X[i : i + 1]

        # Training samples: all original except test sample
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[i] = False
        X_train_orig = X[train_mask]
        y_train_orig = y_encoded[train_mask]

        # Augmented training samples: all except the flipped version of test sample
        test_video_name = video_names[i]
        augmented_test_name = test_video_name + "_flipped"

        # Find and exclude the augmented version of test sample
        aug_mask = video_names_aug != augmented_test_name
        X_train_aug = X_aug[aug_mask]
        y_train_aug = y_aug_encoded[aug_mask]

        # Combine original and augmented training data
        X_train = np.vstack([X_train_orig, X_train_aug])
        y_train = np.concatenate([y_train_orig, y_train_aug])

        # Train SVM
        clf = SVC(kernel=kernel, C=C)
        clf.fit(X_train, y_train)

        # Predict
        predictions[i] = clf.predict(X_test)[0]

        if (i + 1) % 30 == 0:
            print(f"Leave-one-out (augmented): {i + 1}/{n_samples} done...")

    # Calculate accuracy
    accuracy = np.mean(predictions == y_encoded)

    # Decode predictions back to original labels
    predictions_decoded = label_encoder.inverse_transform(predictions)

    print(f"Leave-one-out (augmented) completed: {n_samples}/{n_samples}")

    return accuracy, predictions_decoded, y


def run_classification_experiment_augmented(
    dataset_file,
    augmented_dataset_file,
    vocabulary_path,
    keypoints_dir="data/keypoints",
    descriptor_type="hoghof",
    kernel="rbf",
    C=1.0,
):
    """Run a classification experiment with data augmentation.

    Args:
        dataset_file: Path to original .files dataset file
        augmented_dataset_file: Path to augmented .files dataset file
        vocabulary_path: Path to visual vocabulary .npy file
        keypoints_dir: Directory containing .key files
        descriptor_type: "hoghof", "hog", or "hof"
        kernel: SVM kernel type
        C: SVM regularization parameter

    Returns:
        Tuple of (accuracy, predictions, true_labels, X, y)
    """
    print(f"\n{'=' * 60}")
    print(f"Classification experiment (AUGMENTED): {descriptor_type.upper()}")
    print(f"{'=' * 60}")

    # Load vocabulary
    print(f"Loading vocabulary from {vocabulary_path}...")
    vocabulary = np.load(vocabulary_path)
    print(
        f"Vocabulary size: {vocabulary.shape[0]}, descriptor dim: {vocabulary.shape[1]}"
    )

    # Compute BoVW vectors for original videos
    print(f"\nComputing BoVW vectors for original dataset ({descriptor_type})...")
    X, y, video_names = compute_bovw_dataset(
        dataset_file, vocabulary, keypoints_dir, descriptor_type
    )
    print(f"Original dataset: {X.shape[0]} videos, {X.shape[1]} features")

    # Compute BoVW vectors for augmented videos
    print(f"\nComputing BoVW vectors for augmented dataset ({descriptor_type})...")
    X_aug, y_aug, video_names_aug = compute_bovw_dataset(
        augmented_dataset_file, vocabulary, keypoints_dir, descriptor_type
    )
    print(f"Augmented dataset: {X_aug.shape[0]} videos, {X_aug.shape[1]} features")

    # Get unique classes
    unique_classes = np.unique(y)
    print(f"Classes ({len(unique_classes)}): {list(unique_classes)}")

    # Run leave-one-out classification with augmentation
    print(
        f"\nRunning leave-one-out SVM classification with augmentation (kernel={kernel}, C={C})..."
    )
    accuracy, predictions, true_labels = classify_leave_one_out_augmented(
        X, y, video_names, X_aug, y_aug, video_names_aug, kernel, C
    )

    print(f"\n{'=' * 60}")
    print(
        f"ACCURACY ({descriptor_type.upper()} + AUGMENTATION): {accuracy:.4f} ({accuracy * 100:.2f}%)"
    )
    print(f"{'=' * 60}")

    return accuracy, predictions, true_labels, X, y


def compare_descriptors_augmented(
    dataset_file,
    augmented_dataset_file,
    vocab_dir="visual_vocabularies",
    keypoints_dir="data/keypoints",
    kernel="rbf",
    C=1.0,
):
    """Compare classification performance with data augmentation.

    Args:
        dataset_file: Path to original .files dataset file
        augmented_dataset_file: Path to augmented .files dataset file
        vocab_dir: Directory containing vocabulary files
        keypoints_dir: Directory containing .key files
        kernel: SVM kernel type
        C: SVM regularization parameter

    Returns:
        Dictionary with results for each descriptor type
    """
    results = {}

    descriptor_configs = [
        ("hoghof", f"{vocab_dir}/voc_hoghof_500.npy"),
        ("hog", f"{vocab_dir}/voc_hog_500.npy"),
        ("hof", f"{vocab_dir}/voc_hof_500.npy"),
    ]

    for desc_type, vocab_path in descriptor_configs:
        accuracy, predictions, true_labels, X, y = (
            run_classification_experiment_augmented(
                dataset_file=dataset_file,
                augmented_dataset_file=augmented_dataset_file,
                vocabulary_path=vocab_path,
                keypoints_dir=keypoints_dir,
                descriptor_type=desc_type,
                kernel=kernel,
                C=C,
            )
        )
        results[desc_type] = {
            "accuracy": accuracy,
            "predictions": predictions,
            "true_labels": true_labels,
            "X": X,
            "y": y,
        }

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY - Comparison of descriptors (WITH AUGMENTATION)")
    print(f"{'=' * 60}")
    for desc_type, res in results.items():
        print(
            f"{desc_type.upper():10s}: {res['accuracy']:.4f} ({res['accuracy'] * 100:.2f}%)"
        )

    return results


if __name__ == "__main__":
    """
    # Question 4 - Une seule expérience (HOG+HOF)
    accuracy, predictions, true_labels, X, y = run_classification_experiment(
        dataset_file="data/ucf-sports.files",
        vocabulary_path="visual_vocabularies/voc_hoghof_500.npy",
        keypoints_dir="data/keypoints",
        descriptor_type="hoghof",
    )

    # Question 3a - HOG seul
    accuracy_hog, _, _, _, _ = run_classification_experiment(
        dataset_file="data/ucf-sports.files",
        vocabulary_path="visual_vocabularies/voc_hog_500.npy",
        descriptor_type="hog",
    )

    # Question 3b - HOF seul
    accuracy_hof, _, _, _, _ = run_classification_experiment(
        dataset_file="data/ucf-sports.files",
        vocabulary_path="visual_vocabularies/voc_hof_500.npy",
        descriptor_type="hof",
    )

    # Question 5 - Comparer les trois descripteurs
    results = compare_descriptors(
        dataset_file="data/ucf-sports.files",
        vocab_dir="visual_vocabularies",
        keypoints_dir="data/keypoints",
    )
    """

    # PARTIE 6 - Augmentation de données
    print("\n" + "=" * 60)
    print("PARTIE 6 - AUGMENTATION DE DONNÉES")
    print("=" * 60)

    # Question 6.1 & 6.2 - Classification avec augmentation
    results_augmented = compare_descriptors_augmented(
        dataset_file="data/ucf-sports.files",
        augmented_dataset_file="data/ucf-sports_augmented.files",
        vocab_dir="visual_vocabularies",
        keypoints_dir="data/keypoints",
    )
