import os

import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from descriptor_video.lbp_bovw import compute_bovw_dataset
from descriptor_video.lbp_descriptor import (
    sample_descriptors_for_vocabulary,
)
from descriptor_video.lbp_vocabulary import train_vocabulary
from descriptor_video.stip import read_stip_file


def classify_leave_one_out(X, y, kernel="rbf", C=1.0):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    loo = LeaveOneOut()
    n_samples = len(y_encoded)
    predictions = np.zeros(n_samples, dtype=int)

    for i, (train_idx, test_idx) in enumerate(loo.split(X)):
        clf = SVC(kernel=kernel, C=C)
        clf.fit(X[train_idx], y_encoded[train_idx])
        predictions[test_idx] = clf.predict(X[test_idx])

    accuracy = np.mean(predictions == y_encoded)
    return accuracy


def experiment_vary_kernel(X, y, kernels=("linear", "rbf", "poly"), C=1.0):
    results = {}
    for kernel in kernels:
        acc = classify_leave_one_out(X, y, kernel=kernel, C=C)
        results[kernel] = acc
        print(f"{kernel:<12} {acc * 100:>6.2f}%")
    return results


def experiment_vary_C(X, y, C_values=(0.01, 0.1, 1.0, 10.0, 100.0), kernel="rbf"):
    results = {}
    for C in C_values:
        acc = classify_leave_one_out(X, y, kernel=kernel, C=C)
        results[C] = acc
        print(f"{C:<12} {acc * 100:>6.2f}%")
    return results


def experiment_vary_vocab_size(
    dataset_file,
    descriptor_type="lbp_top",
    vocab_sizes=(100, 250, 500, 1000),
    video_dir="data/videos",
    keypoints_dir="data/keypoints",
    neighborhood_size=3,
    kernel="rbf",
    C=1.0,
):
    sampled_desc = sample_descriptors_for_vocabulary(
        dataset_file=dataset_file,
        video_dir=video_dir,
        keypoints_dir=keypoints_dir,
        neighborhood_size=neighborhood_size,
        descriptor_type=descriptor_type,
        sample_ratio=0.02,
    )

    results = {}
    for vocab_size in vocab_sizes:
        vocabulary = train_vocabulary(sampled_desc, n_clusters=vocab_size)

        X, y, _ = compute_bovw_dataset(
            dataset_file,
            vocabulary,
            video_dir=video_dir,
            keypoints_dir=keypoints_dir,
            neighborhood_size=neighborhood_size,
            descriptor_type=descriptor_type,
        )

        acc = classify_leave_one_out(X, y, kernel=kernel, C=C)
        results[vocab_size] = acc
        print(f"{vocab_size:<12} {acc * 100:>6.2f}%")

    return results


def experiment_vary_neighborhood(
    dataset_file,
    descriptor_type="lbp_top",
    neighborhood_sizes=(3, 5, 7, 9),
    vocab_size=500,
    video_dir="data/videos",
    keypoints_dir="data/keypoints",
    kernel="rbf",
    C=1.0,
):
    results = {}
    for ns in neighborhood_sizes:
        sampled_desc = sample_descriptors_for_vocabulary(
            dataset_file=dataset_file,
            video_dir=video_dir,
            keypoints_dir=keypoints_dir,
            neighborhood_size=ns,
            descriptor_type=descriptor_type,
            sample_ratio=0.02,
        )
        vocabulary = train_vocabulary(sampled_desc, n_clusters=vocab_size)

        X, y, _ = compute_bovw_dataset(
            dataset_file,
            vocabulary,
            video_dir=video_dir,
            keypoints_dir=keypoints_dir,
            neighborhood_size=ns,
            descriptor_type=descriptor_type,
        )

        acc = classify_leave_one_out(X, y, kernel=kernel, C=C)
        results[ns] = acc
        print(f"{ns:<14} {acc * 100:>6.2f}%")

    return results


def run_hyperparameter_search(
    dataset_file="data/ucf-sports.files",
    vocab_dir="visual_vocabularies",
    video_dir="data/videos",
    keypoints_dir="data/keypoints",
):
    all_results = {}

    for desc_type in ["lbp", "lbp_top"]:
        vocab_path = f"{vocab_dir}/voc_{desc_type}_500.npy"
        if not os.path.exists(vocab_path):
            continue

        vocabulary = np.load(vocab_path)
        X, y, _ = compute_bovw_dataset(
            dataset_file,
            vocabulary,
            video_dir=video_dir,
            keypoints_dir=keypoints_dir,
            neighborhood_size=3,
            descriptor_type=desc_type,
        )

        kernel_results = experiment_vary_kernel(X, y)
        c_results = experiment_vary_C(X, y)

        all_results[desc_type] = {
            "kernel": kernel_results,
            "C": c_results,
        }

    return all_results


def analyze_keypoint_scales(
    dataset_file="data/ucf-sports.files",
    keypoints_dir="data/keypoints",
):
    video_names = []
    with open(dataset_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                video_names.append(line.split()[0])

    all_scale_xy = []
    all_scale_t = []

    for video_name in video_names:
        keypoint_file = os.path.join(keypoints_dir, video_name + ".key")
        keypoints, _ = read_stip_file(keypoint_file)
        if keypoints is not None:
            for kp in keypoints:
                _, _, _, scale_xy, scale_t = kp
                all_scale_xy.append(scale_xy)
                all_scale_t.append(scale_t)

    scale_xy = np.array(all_scale_xy)
    scale_t = np.array(all_scale_t)

    print(f"Nombre total de points d'intérêt: {len(scale_xy)}")
    print(
        f"Échelle spatiale  - min: {scale_xy.min():.2f}, max: {scale_xy.max():.2f}, "
        f"mean: {scale_xy.mean():.2f}, median: {np.median(scale_xy):.2f}"
    )
    print(
        f"Échelle temporelle - min: {scale_t.min():.2f}, max: {scale_t.max():.2f}, "
        f"mean: {scale_t.mean():.2f}, median: {np.median(scale_t):.2f}"
    )

    effective_spatial = np.sqrt(scale_xy) * 2
    effective_temporal = np.sqrt(scale_t) * 2

    pct_spatial_too_small = np.mean(effective_spatial > 3) * 100
    pct_temporal_too_small = np.mean(effective_temporal > 3) * 100

    print(
        f"Points où voisinage 3 trop petit - Spatial: {pct_spatial_too_small:.1f}%, "
        f"Temporel: {pct_temporal_too_small:.1f}%"
    )

    return scale_xy, scale_t


def compute_lbp_descriptor_adaptive(
    video,
    y,
    x,
    t,
    scale_xy,
    scale_t,
    target_size=3,
    descriptor_type="lbp_top",
    n_points=8,
    radius=1,
):
    import cv2 as cv2_local
    from skimage.feature import local_binary_pattern

    n_frames, h, w = video.shape

    spatial_size = max(target_size, int(np.ceil(np.sqrt(scale_xy) * 2)))
    temporal_size = max(target_size, int(np.ceil(np.sqrt(scale_t) * 2)))

    if spatial_size % 2 == 0:
        spatial_size += 1
    if temporal_size % 2 == 0:
        temporal_size += 1

    half_s = spatial_size // 2
    half_t = temporal_size // 2

    t = max(half_t, min(t, n_frames - 1 - half_t))
    y = max(half_s, min(y, h - 1 - half_s))
    x = max(half_s, min(x, w - 1 - half_s))

    t_start = max(0, t - half_t)
    t_end = min(n_frames, t + half_t + 1)
    y_start = max(0, y - half_s)
    y_end = min(h, y + half_s + 1)
    x_start = max(0, x - half_s)
    x_end = min(w, x + half_s + 1)

    cube = video[t_start:t_end, y_start:y_end, x_start:x_end]

    resized_slices = []
    for ti in range(cube.shape[0]):
        resized = cv2_local.resize(
            cube[ti], (target_size, target_size), interpolation=cv2_local.INTER_LINEAR
        )
        resized_slices.append(resized)

    resized_cube = np.array(resized_slices, dtype=np.uint8)
    if resized_cube.shape[0] > target_size:
        indices = np.linspace(0, resized_cube.shape[0] - 1, target_size, dtype=int)
        resized_cube = resized_cube[indices]
    elif resized_cube.shape[0] < target_size:
        pad_t = target_size - resized_cube.shape[0]
        resized_cube = np.pad(resized_cube, ((0, pad_t), (0, 0), (0, 0)), mode="edge")

    n_bins = n_points + 2

    def _lbp_hist(plane):
        if plane.shape[0] < 3 or plane.shape[1] < 3:
            pad_y = max(0, 3 - plane.shape[0])
            pad_x = max(0, 3 - plane.shape[1])
            plane = np.pad(plane, ((0, pad_y), (0, pad_x)), mode="edge")
        lbp = local_binary_pattern(plane, n_points, radius, method="uniform")
        hist, _ = np.histogram(
            lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True
        )
        return hist.astype(np.float32)

    if descriptor_type == "lbp":
        mid_t = resized_cube.shape[0] // 2
        return _lbp_hist(resized_cube[mid_t])

    mid_t = resized_cube.shape[0] // 2
    mid_y = resized_cube.shape[1] // 2
    mid_x = resized_cube.shape[2] // 2

    lbp_xy = _lbp_hist(resized_cube[mid_t, :, :])
    lbp_xt = _lbp_hist(resized_cube[:, mid_y, :])
    lbp_yt = _lbp_hist(resized_cube[:, :, mid_x])

    return np.concatenate([lbp_xy, lbp_xt, lbp_yt])


def compute_adaptive_lbp_for_video(
    video_name,
    video_dir="data/videos",
    keypoints_dir="data/keypoints",
    target_size=3,
    descriptor_type="lbp_top",
):
    from descriptor_video.lbp_descriptor import read_video_grayscale

    video_file = os.path.join(video_dir, video_name + ".avi")
    video = read_video_grayscale(video_file)
    if video is None:
        return None, None

    keypoint_file = os.path.join(keypoints_dir, video_name + ".key")
    keypoints, _ = read_stip_file(keypoint_file)
    if keypoints is None:
        return None, None

    descriptors = []
    for kp in keypoints:
        y, x, t, scale_xy, scale_t = kp
        desc = compute_lbp_descriptor_adaptive(
            video,
            y,
            x,
            t,
            scale_xy,
            scale_t,
            target_size=target_size,
            descriptor_type=descriptor_type,
        )
        descriptors.append(desc)

    return keypoints, np.array(descriptors, dtype=np.float32)


def experiment_adaptive_vs_fixed(
    dataset_file="data/ucf-sports.files",
    vocab_dir="visual_vocabularies",
    video_dir="data/videos",
    keypoints_dir="data/keypoints",
    descriptor_type="lbp_top",
    kernel="rbf",
    C=1.0,
):
    from descriptor_video.lbp_bovw import compute_bovw_vector, load_dataset_file

    vocab_path = f"{vocab_dir}/voc_{descriptor_type}_500.npy"
    if not os.path.exists(vocab_path):
        return None

    vocabulary = np.load(vocab_path)

    X_fixed, y_fixed, video_names = compute_bovw_dataset(
        dataset_file,
        vocabulary,
        video_dir=video_dir,
        keypoints_dir=keypoints_dir,
        neighborhood_size=3,
        descriptor_type=descriptor_type,
    )
    acc_fixed = classify_leave_one_out(X_fixed, y_fixed, kernel=kernel, C=C)

    video_names_list, labels = load_dataset_file(dataset_file)
    n_videos = len(video_names_list)
    vocab_size = len(vocabulary)
    X_adaptive = np.zeros((n_videos, vocab_size), dtype=np.float32)

    for i, vname in enumerate(video_names_list):
        _, desc = compute_adaptive_lbp_for_video(
            vname,
            video_dir=video_dir,
            keypoints_dir=keypoints_dir,
            target_size=3,
            descriptor_type=descriptor_type,
        )
        X_adaptive[i] = compute_bovw_vector(desc, vocabulary)

    acc_adaptive = classify_leave_one_out(X_adaptive, labels, kernel=kernel, C=C)

    print(
        f"{descriptor_type.upper()} - Fixed: {acc_fixed * 100:.2f}%, Adaptive: {acc_adaptive * 100:.2f}%"
    )

    return {"fixed": acc_fixed, "adaptive": acc_adaptive}


if __name__ == "__main__":
    hp_results = run_hyperparameter_search(
        dataset_file="data/ucf-sports.files",
        vocab_dir="visual_vocabularies",
        video_dir="data/videos",
        keypoints_dir="data/keypoints",
    )

    scale_xy, scale_t = analyze_keypoint_scales(
        dataset_file="data/ucf-sports.files",
        keypoints_dir="data/keypoints",
    )

    for dt in ["lbp", "lbp_top"]:
        experiment_adaptive_vs_fixed(
            dataset_file="data/ucf-sports.files",
            vocab_dir="visual_vocabularies",
            video_dir="data/videos",
            keypoints_dir="data/keypoints",
            descriptor_type=dt,
        )
