"""
Questions 7.4 et 7.5: Calcul des vecteurs Bag of Visual Words (BoVW)
pour les descripteurs LBP et LBP-TOP.

Ce module calcule les histogrammes de fréquences des mots visuels
pour chaque vidéo à partir des descripteurs LBP/LBP-TOP.
"""

import os

import numpy as np

from descriptor_video.lbp_descriptor import compute_lbp_descriptors_for_video


def compute_bovw_vector(descriptors, vocabulary):
    """
    Question 7.4: Calculer le vecteur de fréquences du sac de mots visuels
    correspondant à une vidéo à partir de ses descripteurs LBP/LBP-TOP.

    Args:
        descriptors: numpy array of shape (n_keypoints, descriptor_dim)
                    containing LBP or LBP-TOP descriptors for a video
        vocabulary: numpy array of shape (vocabulary_size, descriptor_dim)
                   containing the visual words (cluster centers)

    Returns:
        Normalized frequency vector of shape (vocabulary_size,)
    """
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(len(vocabulary), dtype=np.float32)

    vocabulary_size = len(vocabulary)

    # Compute distances from each descriptor to all visual words
    # Using broadcasting: (n_keypoints, 1, dim) - (1, vocab_size, dim)
    distances = np.linalg.norm(
        descriptors[:, np.newaxis, :] - vocabulary[np.newaxis, :, :], axis=2
    )

    # Assign each descriptor to the nearest visual word
    assignments = np.argmin(distances, axis=1)

    # Build histogram of visual word frequencies
    histogram = np.bincount(assignments, minlength=vocabulary_size).astype(np.float32)

    # Normalize histogram (L1 normalization)
    if histogram.sum() > 0:
        histogram = histogram / histogram.sum()

    return histogram


def compute_bovw_for_video(
    video_name,
    vocabulary,
    video_dir="data/videos",
    keypoints_dir="data/keypoints",
    neighborhood_size=3,
    descriptor_type="lbp_top",
):
    """Compute BoVW vector for a single video.

    Args:
        video_name: Name of the video (without extension)
        vocabulary: Visual vocabulary (cluster centers)
        video_dir: Directory containing video files
        keypoints_dir: Directory containing keypoint files
        neighborhood_size: Spatio-temporal neighborhood size
        descriptor_type: "lbp" or "lbp_top"

    Returns:
        Normalized BoVW frequency vector
    """
    # Compute all descriptors for this video
    _, descriptors = compute_lbp_descriptors_for_video(
        video_name,
        video_dir=video_dir,
        keypoints_dir=keypoints_dir,
        neighborhood_size=neighborhood_size,
        descriptor_type=descriptor_type,
    )

    return compute_bovw_vector(descriptors, vocabulary)


def load_dataset_file(dataset_file):
    """Load video names and labels from a .files dataset file.

    Args:
        dataset_file: Path to the .files file

    Returns:
        Tuple of (video_names, labels) as numpy arrays
    """
    video_names = []
    labels = []

    with open(dataset_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            video_names.append(parts[0])
            labels.append(parts[1])

    return np.array(video_names), np.array(labels)


def compute_bovw_dataset(
    dataset_file,
    vocabulary,
    video_dir="data/videos",
    keypoints_dir="data/keypoints",
    neighborhood_size=3,
    descriptor_type="lbp_top",
):
    """
    Question 7.5: Calculer les vecteurs BoVW pour l'ensemble des vidéos
    du jeu de données.

    Args:
        dataset_file: Path to .files dataset file
        vocabulary: Visual vocabulary (cluster centers from K-means)
        video_dir: Directory containing video files
        keypoints_dir: Directory containing .key files
        neighborhood_size: Spatio-temporal neighborhood size
        descriptor_type: "lbp" or "lbp_top"

    Returns:
        Tuple of:
            - X: numpy array of shape (n_videos, vocabulary_size) containing BoVW vectors
            - y: numpy array of shape (n_videos,) containing labels
            - video_names: numpy array of video names
    """
    video_names, labels = load_dataset_file(dataset_file)

    n_videos = len(video_names)
    vocab_size = len(vocabulary)

    X = np.zeros((n_videos, vocab_size), dtype=np.float32)

    print(f"Computing BoVW vectors for {n_videos} videos...")
    print(f"Descriptor type: {descriptor_type.upper()}")
    print(f"Vocabulary size: {vocab_size}")

    for i, video_name in enumerate(video_names):
        X[i] = compute_bovw_for_video(
            video_name,
            vocabulary,
            video_dir=video_dir,
            keypoints_dir=keypoints_dir,
            neighborhood_size=neighborhood_size,
            descriptor_type=descriptor_type,
        )

        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{n_videos} videos...")

    print(f"Processed {n_videos} videos.")

    return X, labels, video_names


if __name__ == "__main__":
    print("=" * 60)
    print("Questions 7.4 & 7.5: Calcul des vecteurs BoVW")
    print("=" * 60)

    # ================================================================
    # Test avec LBP
    # ================================================================
    print("\n--- Test LBP ---")

    # Charger le vocabulaire LBP (doit être créé avec lbp_vocabulary.py)
    lbp_vocab_path = "visual_vocabularies/voc_lbp_500.npy"
    if os.path.exists(lbp_vocab_path):
        lbp_vocabulary = np.load(lbp_vocab_path)
        print(f"Loaded LBP vocabulary: {lbp_vocabulary.shape}")

        # Calculer BoVW pour toutes les vidéos
        X_lbp, y_lbp, video_names = compute_bovw_dataset(
            dataset_file="data/ucf-sports.files",
            vocabulary=lbp_vocabulary,
            video_dir="data/videos",
            keypoints_dir="data/keypoints",
            neighborhood_size=3,
            descriptor_type="lbp",
        )
        print(f"\nLBP BoVW matrix shape: {X_lbp.shape}")
        print(f"Labels shape: {y_lbp.shape}")
    else:
        print(f"Vocabulary not found: {lbp_vocab_path}")
        print("Run lbp_vocabulary.py first to create the vocabulary.")

    # ================================================================
    # Test avec LBP-TOP
    # ================================================================
    print("\n--- Test LBP-TOP ---")

    # Charger le vocabulaire LBP-TOP
    lbp_top_vocab_path = "visual_vocabularies/voc_lbp_top_500.npy"
    if os.path.exists(lbp_top_vocab_path):
        lbp_top_vocabulary = np.load(lbp_top_vocab_path)
        print(f"Loaded LBP-TOP vocabulary: {lbp_top_vocabulary.shape}")

        # Calculer BoVW pour toutes les vidéos
        X_lbp_top, y_lbp_top, video_names = compute_bovw_dataset(
            dataset_file="data/ucf-sports.files",
            vocabulary=lbp_top_vocabulary,
            video_dir="data/videos",
            keypoints_dir="data/keypoints",
            neighborhood_size=3,
            descriptor_type="lbp_top",
        )
        print(f"\nLBP-TOP BoVW matrix shape: {X_lbp_top.shape}")
        print(f"Labels shape: {y_lbp_top.shape}")
    else:
        print(f"Vocabulary not found: {lbp_top_vocab_path}")
        print("Run lbp_vocabulary.py first to create the vocabulary.")
