"""
Question 7.3: Entraînement du vocabulaire visuel pour LBP et LBP-TOP.

Ce module entraîne un K-Means sur les descripteurs LBP/LBP-TOP échantillonnés
et sauvegarde les centres des clusters (mots visuels) au format numpy.
"""

import os

import numpy as np
from sklearn.cluster import KMeans

from descriptor_video.lbp_descriptor import sample_descriptors_for_vocabulary


def train_vocabulary(
    descriptors,
    vocabulary_size=500,
    random_seed=42,
    n_init=10,
    max_iter=300,
    verbose=True,
):
    """Train a visual vocabulary using K-Means clustering.

    Args:
        descriptors: numpy array of shape (n_samples, descriptor_dim)
        vocabulary_size: Number of visual words (clusters)
        random_seed: Random seed for reproducibility
        n_init: Number of K-Means initializations
        max_iter: Maximum iterations per initialization
        verbose: Print progress information

    Returns:
        vocabulary: numpy array of shape (vocabulary_size, descriptor_dim)
                   containing the cluster centers (visual words)
    """
    if verbose:
        print(f"Training K-Means with {vocabulary_size} clusters...")
        print(f"Input: {descriptors.shape[0]} descriptors, {descriptors.shape[1]} dimensions")

    kmeans = KMeans(
        n_clusters=vocabulary_size,
        random_state=random_seed,
        n_init=n_init,
        max_iter=max_iter,
        verbose=1 if verbose else 0,
    )

    kmeans.fit(descriptors)

    if verbose:
        print(f"K-Means converged. Inertia: {kmeans.inertia_:.2f}")

    return kmeans.cluster_centers_.astype(np.float32)


def save_vocabulary(vocabulary, output_path):
    """Save vocabulary to a numpy file.

    Args:
        vocabulary: numpy array of cluster centers
        output_path: Path to save the .npy file
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.save(output_path, vocabulary)
    print(f"Vocabulary saved to: {output_path}")
    print(f"Shape: {vocabulary.shape}")


def build_lbp_vocabularies(
    dataset_file="data/ucf-sports.files",
    video_dir="data/videos",
    keypoints_dir="data/keypoints",
    output_dir="visual_vocabularies",
    vocabulary_size=500,
    sample_ratio=0.02,
    neighborhood_size=3,
    random_seed=42,
):
    """Build and save LBP and LBP-TOP vocabularies.

    Args:
        dataset_file: Path to .files dataset file
        video_dir: Directory containing video files
        keypoints_dir: Directory containing keypoint files
        output_dir: Directory to save vocabulary files
        vocabulary_size: Number of visual words
        sample_ratio: Fraction of keypoints to sample (default 2%)
        neighborhood_size: Spatio-temporal neighborhood size
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with 'lbp' and 'lbp_top' vocabularies
    """
    vocabularies = {}

    # ================================================================
    # LBP Vocabulary
    # ================================================================
    print("\n" + "=" * 60)
    print("Building LBP Vocabulary")
    print("=" * 60)

    # Sample descriptors
    print("\nStep 1: Sampling LBP descriptors...")
    lbp_descriptors = sample_descriptors_for_vocabulary(
        dataset_file=dataset_file,
        video_dir=video_dir,
        keypoints_dir=keypoints_dir,
        neighborhood_size=neighborhood_size,
        descriptor_type="lbp",
        sample_ratio=sample_ratio,
        random_seed=random_seed,
    )

    # Train K-Means
    print("\nStep 2: Training K-Means...")
    lbp_vocabulary = train_vocabulary(
        lbp_descriptors,
        vocabulary_size=vocabulary_size,
        random_seed=random_seed,
    )

    # Save vocabulary
    lbp_output_path = os.path.join(output_dir, f"voc_lbp_{vocabulary_size}.npy")
    save_vocabulary(lbp_vocabulary, lbp_output_path)
    vocabularies["lbp"] = lbp_vocabulary

    # ================================================================
    # LBP-TOP Vocabulary
    # ================================================================
    print("\n" + "=" * 60)
    print("Building LBP-TOP Vocabulary")
    print("=" * 60)

    # Sample descriptors
    print("\nStep 1: Sampling LBP-TOP descriptors...")
    lbp_top_descriptors = sample_descriptors_for_vocabulary(
        dataset_file=dataset_file,
        video_dir=video_dir,
        keypoints_dir=keypoints_dir,
        neighborhood_size=neighborhood_size,
        descriptor_type="lbp_top",
        sample_ratio=sample_ratio,
        random_seed=random_seed,
    )

    # Train K-Means
    print("\nStep 2: Training K-Means...")
    lbp_top_vocabulary = train_vocabulary(
        lbp_top_descriptors,
        vocabulary_size=vocabulary_size,
        random_seed=random_seed,
    )

    # Save vocabulary
    lbp_top_output_path = os.path.join(output_dir, f"voc_lbp_top_{vocabulary_size}.npy")
    save_vocabulary(lbp_top_vocabulary, lbp_top_output_path)
    vocabularies["lbp_top"] = lbp_top_vocabulary

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"LBP vocabulary:     {lbp_output_path}")
    print(f"  - Shape: {lbp_vocabulary.shape}")
    print(f"LBP-TOP vocabulary: {lbp_top_output_path}")
    print(f"  - Shape: {lbp_top_vocabulary.shape}")

    return vocabularies


if __name__ == "__main__":
    # Build vocabularies for LBP and LBP-TOP
    vocabularies = build_lbp_vocabularies(
        dataset_file="data/ucf-sports.files",
        video_dir="data/videos",
        keypoints_dir="data/keypoints",
        output_dir="visual_vocabularies",
        vocabulary_size=500,
        sample_ratio=0.02,  # 2% des points d'intérêt
        neighborhood_size=3,  # Taille du voisinage spatio-temporel
        random_seed=42,
    )
