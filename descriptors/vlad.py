import os

import cv2
import numpy as np
import sklearn.neighbors as skln
from sklearn.decomposition import PCA

from bow import sift_kp_desc


def vlad(
    descriptors,
    vocabulary_path="vocabularies_sift/vocabulary_5000.npy",
    use_l2_norm=True,
    use_sqrt_norm=True,
):
    vocabulary = np.load(vocabulary_path)
    vlad = np.zeros(vocabulary.shape, dtype=np.float64)
    quantizer = skln.NearestNeighbors(n_neighbors=1, algorithm="brute").fit(vocabulary)
    ws = quantizer.kneighbors(descriptors, return_distance=False).reshape(-1)

    for i in range(len(vlad)):
        if (ws == i).any():
            vlad[i, :] = np.sum(descriptors[ws == i] - vocabulary[i], axis=0)

    if use_sqrt_norm:
        vlad[:] = np.sign(vlad) * np.sqrt(np.abs(vlad))

    vlad = vlad.reshape((vlad.shape[0] * vlad.shape[1],))
    if use_l2_norm:
        vlad[:] = vlad / np.maximum(np.linalg.norm(vlad), 1e-12)

    return vlad


def vlad_pca(vlad_vectors, n_components=100):
    mean_vector = np.mean(vlad_vectors, axis=0)
    centered_vectors = vlad_vectors - mean_vector

    pca = PCA(n_components=n_components)
    reduced_vectors = pca.fit_transform(centered_vectors)

    return reduced_vectors, pca, mean_vector


def vlad_pca_transform(vlad_vectors, pca_model, mean_vector):
    centered_vectors = vlad_vectors - mean_vector
    return pca_model.transform(centered_vectors)


def dataset_to_vlad(folder_dir):
    vlads = {}

    for image_path in os.listdir(folder_dir):
        image = cv2.imread(os.path.join(folder_dir, image_path))
        kp, desc = sift_kp_desc(image)
        vlads[image_path] = vlad(desc)

    return vlads


def dataset_to_vlad_pca(folder_dir, n_components=100):
    vlads_raw = dataset_to_vlad(folder_dir)
    image_names = list(vlads_raw.keys())
    vlad_matrix = np.array([vlads_raw[name] for name in image_names])
    reduced_matrix, pca_model, mean_vector = vlad_pca(vlad_matrix, n_components)

    vlads_dict = {}
    for i, name in enumerate(image_names):
        vlads_dict[name] = reduced_matrix[i]

    return vlads_dict
