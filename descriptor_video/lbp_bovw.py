
import os

import numpy as np

from descriptor_video.lbp_descriptor import compute_lbp_descriptors_for_video


def compute_bovw_vector(descriptors, vocabulary):
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(len(vocabulary), dtype=np.float32)

    vocabulary_size = len(vocabulary)

    distances = np.linalg.norm(
        descriptors[:, np.newaxis, :] - vocabulary[np.newaxis, :, :], axis=2
    )

    assignments = np.argmin(distances, axis=1)

    histogram = np.bincount(assignments, minlength=vocabulary_size).astype(np.float32)

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
    _, descriptors = compute_lbp_descriptors_for_video(
        video_name,
        video_dir=video_dir,
        keypoints_dir=keypoints_dir,
        neighborhood_size=neighborhood_size,
        descriptor_type=descriptor_type,
    )

    return compute_bovw_vector(descriptors, vocabulary)


def load_dataset_file(dataset_file):
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
    print("Questions 7.4 & 7.5: Calcul des vecteurs BoVW")

    print("\n--- Test LBP ---")

    lbp_vocab_path = "visual_vocabularies/voc_lbp_500.npy"
    if os.path.exists(lbp_vocab_path):
        lbp_vocabulary = np.load(lbp_vocab_path)
        print(f"Loaded LBP vocabulary: {lbp_vocabulary.shape}")

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

    print("\n--- Test LBP-TOP ---")

    lbp_top_vocab_path = "visual_vocabularies/voc_lbp_top_500.npy"
    if os.path.exists(lbp_top_vocab_path):
        lbp_top_vocabulary = np.load(lbp_top_vocab_path)
        print(f"Loaded LBP-TOP vocabulary: {lbp_top_vocabulary.shape}")

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
