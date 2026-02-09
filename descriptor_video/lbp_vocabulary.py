
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
    vocabularies = {}

    print("Building LBP Vocabulary")

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

    print("\nStep 2: Training K-Means...")
    lbp_vocabulary = train_vocabulary(
        lbp_descriptors,
        vocabulary_size=vocabulary_size,
        random_seed=random_seed,
    )

    lbp_output_path = os.path.join(output_dir, f"voc_lbp_{vocabulary_size}.npy")
    save_vocabulary(lbp_vocabulary, lbp_output_path)
    vocabularies["lbp"] = lbp_vocabulary

    print("Building LBP-TOP Vocabulary")

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

    print("\nStep 2: Training K-Means...")
    lbp_top_vocabulary = train_vocabulary(
        lbp_top_descriptors,
        vocabulary_size=vocabulary_size,
        random_seed=random_seed,
    )

    lbp_top_output_path = os.path.join(output_dir, f"voc_lbp_top_{vocabulary_size}.npy")
    save_vocabulary(lbp_top_vocabulary, lbp_top_output_path)
    vocabularies["lbp_top"] = lbp_top_vocabulary

    print("SUMMARY")
    print(f"LBP vocabulary:     {lbp_output_path}")
    print(f"  - Shape: {lbp_vocabulary.shape}")
    print(f"LBP-TOP vocabulary: {lbp_top_output_path}")
    print(f"  - Shape: {lbp_top_vocabulary.shape}")

    return vocabularies


if __name__ == "__main__":
    vocabularies = build_lbp_vocabularies(
        dataset_file="data/ucf-sports.files",
        video_dir="data/videos",
        keypoints_dir="data/keypoints",
        output_dir="visual_vocabularies",
        vocabulary_size=500,
        sample_ratio=0.02,
        neighborhood_size=3,
        random_seed=42,
    )
