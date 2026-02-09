import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from descriptor_video.stip import read_stip_file


def read_video(video_file):
    capture = cv2.VideoCapture(video_file)
    frames = []
    ok, frame = capture.read()
    while ok:
        frames.append(frame)
        ok, frame = capture.read()
    capture.release()
    return frames


def dataset_to_key_desc(folder_dir):
    descriptors = {}

    for keypoint_path in os.listdir(folder_dir):
        keypoints, desc = read_stip_file(os.path.join(folder_dir, keypoint_path))
        descriptors[keypoint_path] = keypoints, desc

    return descriptors


def visualize_keypoints(
    video_name, video_dir="data/videos", keypoints_dir="data/keypoints", output_dir=None
):
    keypoint_file = os.path.join(keypoints_dir, video_name + ".key")
    keypoints, _ = read_stip_file(keypoint_file)

    if keypoints is None:
        print(f"No keypoints found for {video_name}")
        return []

    video_file = os.path.join(video_dir, video_name + ".avi")
    frames = read_video(video_file)

    if not frames:
        print(f"Could not read video {video_file}")
        return []

    keypoints_by_frame = {}
    for kp in keypoints:
        y, x, t, sigma2, tau2 = kp
        if t not in keypoints_by_frame:
            keypoints_by_frame[t] = []
        keypoints_by_frame[t].append((x, y, sigma2))

    annotated_frames = []
    for frame_idx, frame in enumerate(frames):
        frame_copy = frame.copy()

        if frame_idx in keypoints_by_frame:
            for x, y, sigma2 in keypoints_by_frame[frame_idx]:
                radius = int(np.sqrt(sigma2) * 2)
                radius = max(radius, 3)
                cv2.circle(frame_copy, (x, y), radius, (0, 255, 0), 2)
                cv2.circle(frame_copy, (x, y), 2, (0, 0, 255), -1)

        annotated_frames.append(frame_copy)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for i, frame in enumerate(annotated_frames):
            output_path = os.path.join(output_dir, f"{video_name}_frame_{i:04d}.png")
            cv2.imwrite(output_path, frame)
        print(f"Saved {len(annotated_frames)} frames to {output_dir}")
    else:
        frames_with_kp = [i for i in keypoints_by_frame.keys() if i < len(frames)]
        if frames_with_kp:
            sample_frames = sorted(frames_with_kp)[: min(6, len(frames_with_kp))]
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            for idx, frame_idx in enumerate(sample_frames):
                if idx < len(axes):
                    rgb_frame = cv2.cvtColor(
                        annotated_frames[frame_idx], cv2.COLOR_BGR2RGB
                    )
                    axes[idx].imshow(rgb_frame)
                    axes[idx].set_title(
                        f"Frame {frame_idx} ({len(keypoints_by_frame[frame_idx])} keypoints)"
                    )
                    axes[idx].axis("off")

            for idx in range(len(sample_frames), len(axes)):
                axes[idx].axis("off")

            plt.suptitle(f"Keypoints visualization: {video_name}")
            plt.tight_layout()
            plt.show()

    return annotated_frames


def compute_bovw_vector(descriptors, vocabulary):
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(len(vocabulary), dtype=np.float32)

    vocabulary_size = len(vocabulary)

    distances = np.linalg.norm(
        descriptors[:, np.newaxis, :] - vocabulary[np.newaxis, :, :],
        axis=2
    )

    assignments = np.argmin(distances, axis=1)

    histogram = np.bincount(assignments, minlength=vocabulary_size).astype(np.float32)

    if histogram.sum() > 0:
        histogram = histogram / histogram.sum()

    return histogram


def compute_bovw_vector_for_video(video_name, vocabulary, keypoints_dir="data/keypoints",
                                   descriptor_type="hoghof"):
    keypoint_file = os.path.join(keypoints_dir, video_name + ".key")
    _, descriptors = read_stip_file(keypoint_file)

    if descriptors is None:
        return np.zeros(len(vocabulary), dtype=np.float32)

    if descriptor_type == "hog":
        descriptors = descriptors[:, :72]
    elif descriptor_type == "hof":
        descriptors = descriptors[:, 72:]

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


def compute_bovw_dataset(dataset_file, vocabulary, keypoints_dir="data/keypoints",
                         descriptor_type="hoghof"):
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
    visualize_keypoints("Diving-Side_001")

    visualize_keypoints("Diving-Side_001", output_dir="output/keypoints_viz")

    visualize_keypoints(
        "Golf-Swing-Front_001", video_dir="data/videos", keypoints_dir="data/keypoints"
    )
