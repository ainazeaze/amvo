
import os

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from descriptor_video.stip import read_stip_file


def read_video_grayscale(video_file):
    capture = cv2.VideoCapture(video_file)
    frames = []
    ok, frame = capture.read()
    while ok:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        ok, frame = capture.read()
    capture.release()

    if len(frames) == 0:
        return None
    return np.array(frames, dtype=np.uint8)


def compute_lbp_histogram(image, n_points=8, radius=1, method="uniform"):
    lbp = local_binary_pattern(image, n_points, radius, method=method)

    if method == "uniform":
        n_bins = n_points + 2
    else:
        n_bins = 2**n_points

    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def extract_lbp_descriptor(video, y, x, t, neighborhood_size=3, n_points=8, radius=1):
    n_frames, h, w = video.shape
    half_size = neighborhood_size // 2

    t = max(0, min(t, n_frames - 1))

    y_start = max(0, y - half_size)
    y_end = min(h, y + half_size + 1)
    x_start = max(0, x - half_size)
    x_end = min(w, x + half_size + 1)

    patch = video[t, y_start:y_end, x_start:x_end]

    if patch.shape[0] < 3 or patch.shape[1] < 3:
        pad_y = max(0, 3 - patch.shape[0])
        pad_x = max(0, 3 - patch.shape[1])
        patch = np.pad(patch, ((0, pad_y), (0, pad_x)), mode="edge")

    return compute_lbp_histogram(patch, n_points, radius, method="uniform")


def extract_lbp_top_descriptor(
    video, y, x, t, neighborhood_size=3, n_points=8, radius=1
):
    n_frames, h, w = video.shape
    half_size = neighborhood_size // 2

    t = max(half_size, min(t, n_frames - 1 - half_size))
    y = max(half_size, min(y, h - 1 - half_size))
    x = max(half_size, min(x, w - 1 - half_size))

    t_start = max(0, t - half_size)
    t_end = min(n_frames, t + half_size + 1)
    y_start = max(0, y - half_size)
    y_end = min(h, y + half_size + 1)
    x_start = max(0, x - half_size)
    x_end = min(w, x + half_size + 1)

    cube = video[t_start:t_end, y_start:y_end, x_start:x_end]

    if cube.shape[0] < 3:
        pad_t = 3 - cube.shape[0]
        cube = np.pad(cube, ((0, pad_t), (0, 0), (0, 0)), mode="edge")
    if cube.shape[1] < 3:
        pad_y = 3 - cube.shape[1]
        cube = np.pad(cube, ((0, 0), (0, pad_y), (0, 0)), mode="edge")
    if cube.shape[2] < 3:
        pad_x = 3 - cube.shape[2]
        cube = np.pad(cube, ((0, 0), (0, 0), (0, pad_x)), mode="edge")

    mid_t = cube.shape[0] // 2
    xy_plane = cube[mid_t, :, :]
    lbp_xy = compute_lbp_histogram(xy_plane, n_points, radius, method="uniform")

    mid_y = cube.shape[1] // 2
    xt_plane = cube[:, mid_y, :]
    lbp_xt = compute_lbp_histogram(xt_plane, n_points, radius, method="uniform")

    mid_x = cube.shape[2] // 2
    yt_plane = cube[:, :, mid_x]
    lbp_yt = compute_lbp_histogram(yt_plane, n_points, radius, method="uniform")

    return np.concatenate([lbp_xy, lbp_xt, lbp_yt])


def compute_lbp_descriptors_for_video(
    video_name,
    video_dir="data/videos",
    keypoints_dir="data/keypoints",
    neighborhood_size=3,
    descriptor_type="lbp_top",
):
    video_file = os.path.join(video_dir, video_name + ".avi")
    video = read_video_grayscale(video_file)

    if video is None:
        print(f"Could not read video: {video_file}")
        return None, None

    keypoint_file = os.path.join(keypoints_dir, video_name + ".key")
    keypoints, _ = read_stip_file(keypoint_file)

    if keypoints is None:
        print(f"No keypoints found for: {video_name}")
        return None, None

    descriptors = []
    for kp in keypoints:
        y, x, t, sigma2, tau2 = kp

        if descriptor_type == "lbp":
            desc = extract_lbp_descriptor(video, y, x, t, neighborhood_size)
        else:
            desc = extract_lbp_top_descriptor(video, y, x, t, neighborhood_size)

        descriptors.append(desc)

    return keypoints, np.array(descriptors, dtype=np.float32)


def compute_all_lbp_descriptors(
    dataset_file,
    video_dir="data/videos",
    keypoints_dir="data/keypoints",
    neighborhood_size=3,
    descriptor_type="lbp_top",
):
    video_names = []
    labels = []
    with open(dataset_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                video_names.append(parts[0])
                labels.append(parts[1])

    all_descriptors = {}

    for i, video_name in enumerate(video_names):
        keypoints, descriptors = compute_lbp_descriptors_for_video(
            video_name, video_dir, keypoints_dir, neighborhood_size, descriptor_type
        )

        if descriptors is not None:
            all_descriptors[video_name] = {
                "keypoints": keypoints,
                "descriptors": descriptors,
                "label": labels[i],
            }

        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{len(video_names)} videos...")

    print(f"Processed {len(video_names)} videos.")
    return all_descriptors


def sample_descriptors_for_vocabulary(
    dataset_file,
    video_dir="data/videos",
    keypoints_dir="data/keypoints",
    neighborhood_size=3,
    descriptor_type="lbp_top",
    sample_ratio=0.02,
    random_seed=42,
):
    np.random.seed(random_seed)

    video_names = []
    with open(dataset_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                video_names.append(parts[0])

    all_sampled_descriptors = []
    total_keypoints = 0
    total_sampled = 0

    print(f"Sampling {sample_ratio * 100:.1f}% of keypoints from each video...")
    print(f"Descriptor type: {descriptor_type.upper()}")
    print(f"Neighborhood size: {neighborhood_size}")

    for i, video_name in enumerate(video_names):
        keypoints, descriptors = compute_lbp_descriptors_for_video(
            video_name, video_dir, keypoints_dir, neighborhood_size, descriptor_type
        )

        if descriptors is None or len(descriptors) == 0:
            continue

        n_keypoints = len(descriptors)
        total_keypoints += n_keypoints

        n_samples = max(1, int(n_keypoints * sample_ratio))
        indices = np.random.choice(n_keypoints, size=n_samples, replace=False)

        sampled_desc = descriptors[indices]
        all_sampled_descriptors.append(sampled_desc)
        total_sampled += n_samples

        if (i + 1) % 30 == 0:
            print(f"Processed {i + 1}/{len(video_names)} videos...")

    print(f"\nProcessed {len(video_names)} videos.")
    print(f"Total keypoints: {total_keypoints}")
    print(f"Sampled keypoints: {total_sampled} ({total_sampled / total_keypoints * 100:.2f}%)")

    all_descriptors = np.vstack(all_sampled_descriptors)
    print(f"Final descriptor matrix shape: {all_descriptors.shape}")

    return all_descriptors


if __name__ == "__main__":
    video_name = "Diving-Side_001"

    print("Test des descripteurs LBP et LBP-TOP")

    print("\n--- LBP (2D) ---")
    keypoints, lbp_desc = compute_lbp_descriptors_for_video(
        video_name, descriptor_type="lbp", neighborhood_size=3
    )
    if lbp_desc is not None:
        print(f"Nombre de keypoints: {len(keypoints)}")
        print(f"Dimension du descripteur LBP: {lbp_desc.shape[1]}")
        print(f"Shape totale: {lbp_desc.shape}")

    print("\n--- LBP-TOP (3D) ---")
    keypoints, lbp_top_desc = compute_lbp_descriptors_for_video(
        video_name, descriptor_type="lbp_top", neighborhood_size=3
    )
    if lbp_top_desc is not None:
        print(f"Nombre de keypoints: {len(keypoints)}")
        print(f"Dimension du descripteur LBP-TOP: {lbp_top_desc.shape[1]}")
        print(f"Shape totale: {lbp_top_desc.shape}")
        print(f"  - LBP XY: 10 bins (uniform)")
        print(f"  - LBP XT: 10 bins (uniform)")
        print(f"  - LBP YT: 10 bins (uniform)")
        print(f"  - Total: 30 bins")

    print("Question 7.2: Échantillonnage pour vocabulaire")

    print("\n--- Échantillonnage LBP ---")
    lbp_samples = sample_descriptors_for_vocabulary(
        dataset_file="data/ucf-sports.files",
        video_dir="data/videos",
        keypoints_dir="data/keypoints",
        neighborhood_size=3,
        descriptor_type="lbp",
        sample_ratio=0.02,
    )

    print("\n--- Échantillonnage LBP-TOP ---")
    lbp_top_samples = sample_descriptors_for_vocabulary(
        dataset_file="data/ucf-sports.files",
        video_dir="data/videos",
        keypoints_dir="data/keypoints",
        neighborhood_size=3,
        descriptor_type="lbp_top",
        sample_ratio=0.02,
    )
