import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from descriptor_video.stip import read_stip_file


def read_video(video_file):
    """Read video frames from file."""
    capture = cv2.VideoCapture(video_file)
    frames = []
    ok, frame = capture.read()
    while ok:
        frames.append(frame)
        ok, frame = capture.read()
    capture.release()
    return frames


def dataset_to_key_desc(folder_dir):
    """Load keypoints and descriptors for each videos in folder_dir dataset

    Args:
        folder_dir: Path to the directory containing the images.

    Returns:
        Dictionary mapping image filenames to their computed descriptors.
    """
    descriptors = {}

    for keypoint_path in os.listdir(folder_dir):
        keypoints, desc = read_stip_file(os.path.join(folder_dir, keypoint_path))
        descriptors[keypoint_path] = keypoints, desc

    return descriptors


def visualize_keypoints(
    video_name, video_dir="data/videos", keypoints_dir="data/keypoints", output_dir=None
):
    """Visualize keypoints detected in a video.

    Args:
        video_name: Name of the video file (without extension, e.g., "Diving-Side_001")
        video_dir: Directory containing video files
        keypoints_dir: Directory containing keypoint files
        output_dir: If provided, save frames to this directory instead of displaying

    Returns:
        List of frames with keypoints drawn on them
    """
    # Load keypoints
    keypoint_file = os.path.join(keypoints_dir, video_name + ".key")
    keypoints, _ = read_stip_file(keypoint_file)

    if keypoints is None:
        print(f"No keypoints found for {video_name}")
        return []

    # Load video
    video_file = os.path.join(video_dir, video_name + ".avi")
    frames = read_video(video_file)

    if not frames:
        print(f"Could not read video {video_file}")
        return []

    # Group keypoints by frame
    keypoints_by_frame = {}
    for kp in keypoints:
        y, x, t, sigma2, tau2 = kp
        if t not in keypoints_by_frame:
            keypoints_by_frame[t] = []
        keypoints_by_frame[t].append((x, y, sigma2))

    # Draw keypoints on frames
    annotated_frames = []
    for frame_idx, frame in enumerate(frames):
        frame_copy = frame.copy()

        if frame_idx in keypoints_by_frame:
            for x, y, sigma2 in keypoints_by_frame[frame_idx]:
                # Radius based on spatial scale
                radius = int(np.sqrt(sigma2) * 2)
                radius = max(radius, 3)
                # Draw circle at keypoint location
                cv2.circle(frame_copy, (x, y), radius, (0, 255, 0), 2)
                # Draw center point
                cv2.circle(frame_copy, (x, y), 2, (0, 0, 255), -1)

        annotated_frames.append(frame_copy)

    # Save or display
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for i, frame in enumerate(annotated_frames):
            output_path = os.path.join(output_dir, f"{video_name}_frame_{i:04d}.png")
            cv2.imwrite(output_path, frame)
        print(f"Saved {len(annotated_frames)} frames to {output_dir}")
    else:
        # Display a sample of frames with keypoints
        frames_with_kp = [i for i in keypoints_by_frame.keys() if i < len(frames)]
        if frames_with_kp:
            sample_frames = sorted(frames_with_kp)[: min(6, len(frames_with_kp))]
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            for idx, frame_idx in enumerate(sample_frames):
                if idx < len(axes):
                    # Convert BGR to RGB for matplotlib
                    rgb_frame = cv2.cvtColor(
                        annotated_frames[frame_idx], cv2.COLOR_BGR2RGB
                    )
                    axes[idx].imshow(rgb_frame)
                    axes[idx].set_title(
                        f"Frame {frame_idx} ({len(keypoints_by_frame[frame_idx])} keypoints)"
                    )
                    axes[idx].axis("off")

            # Hide unused subplots
            for idx in range(len(sample_frames), len(axes)):
                axes[idx].axis("off")

            plt.suptitle(f"Keypoints visualization: {video_name}")
            plt.tight_layout()
            plt.show()

    return annotated_frames


def compute_bovw_vector(descriptors, vocabulary):
    """Compute the Bag of Visual Words frequency vector for a video.

    This function assigns each local descriptor to its nearest visual word
    in the vocabulary and computes a normalized histogram of visual word frequencies.

    Args:
        descriptors: numpy array of shape (n_keypoints, descriptor_dim) containing
                     the local descriptors (HOG+HOF) for a video. For HOG+HOF,
                     descriptor_dim = 162 (72 HOG + 90 HOF).
        vocabulary: numpy array of shape (vocabulary_size, descriptor_dim) containing
                    the visual words (cluster centers from K-means).

    Returns:
        Normalized frequency vector of shape (vocabulary_size,) representing
        the Bag of Visual Words histogram for the video.
    """
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(len(vocabulary), dtype=np.float32)

    vocabulary_size = len(vocabulary)

    # Compute distances from each descriptor to all visual words
    # Using broadcasting: (n_keypoints, 1, dim) - (1, vocab_size, dim)
    # Result: (n_keypoints, vocab_size)
    distances = np.linalg.norm(
        descriptors[:, np.newaxis, :] - vocabulary[np.newaxis, :, :],
        axis=2
    )

    # Assign each descriptor to the nearest visual word
    assignments = np.argmin(distances, axis=1)

    # Build histogram of visual word frequencies
    histogram = np.bincount(assignments, minlength=vocabulary_size).astype(np.float32)

    # Normalize histogram (L1 normalization)
    if histogram.sum() > 0:
        histogram = histogram / histogram.sum()

    return histogram


def compute_bovw_vector_for_video(video_name, vocabulary, keypoints_dir="data/keypoints",
                                   descriptor_type="hoghof"):
    """Compute BoVW vector for a single video by its name.

    Args:
        video_name: Name of the video (without extension)
        vocabulary: Visual vocabulary (cluster centers)
        keypoints_dir: Directory containing .key files
        descriptor_type: Type of descriptor to use ("hoghof", "hog", or "hof")

    Returns:
        Normalized BoVW frequency vector
    """
    keypoint_file = os.path.join(keypoints_dir, video_name + ".key")
    _, descriptors = read_stip_file(keypoint_file)

    if descriptors is None:
        return np.zeros(len(vocabulary), dtype=np.float32)

    # Select descriptor components based on type
    if descriptor_type == "hog":
        descriptors = descriptors[:, :72]  # First 72 components (HOG)
    elif descriptor_type == "hof":
        descriptors = descriptors[:, 72:]  # Last 90 components (HOF)
    # else: use full HOG+HOF (162 components)

    return compute_bovw_vector(descriptors, vocabulary)


def load_dataset_file(dataset_file):
    """Load video names and labels from a .files dataset file.

    Args:
        dataset_file: Path to the .files file (e.g., "data/ucf-sports.files")

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


def compute_bovw_dataset(dataset_file, vocabulary, keypoints_dir="data/keypoints",
                         descriptor_type="hoghof"):
    """Compute BoVW vectors for all videos in a dataset.

    Args:
        dataset_file: Path to the .files file containing video names and labels
        vocabulary: Visual vocabulary (cluster centers from K-means)
        keypoints_dir: Directory containing .key files
        descriptor_type: Type of descriptor to use ("hoghof", "hog", or "hof")

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

    for i, video_name in enumerate(video_names):
        X[i] = compute_bovw_vector_for_video(
            video_name, vocabulary, keypoints_dir, descriptor_type
        )
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{n_videos} videos...")

    print(f"Processed {n_videos} videos.")

    return X, labels, video_names


if __name__ == "__main__":
    # Display sample frames with keypoints
    visualize_keypoints("Diving-Side_001")

    # Save all annotated frames to a directory
    visualize_keypoints("Diving-Side_001", output_dir="output/keypoints_viz")

    # With custom paths
    visualize_keypoints(
        "Golf-Swing-Front_001", video_dir="data/videos", keypoints_dir="data/keypoints"
    )
