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
    """Compute the VLAD descriptors of an image.

    @param sifts SIFT descriptors extracted from an image
    @param vocabulary Visual vocabulary
    @param use_l2_norm True to use global L2 normalization, False otherwise (default: True)
    @param use_sqrt_norm True to use square root normlization, False otherwise (default: True)
    @type sifts Array of shape (N, 128) (N = number of descriptors in the image)
    @type vocabulary Numpy array of shape (K, 128)
    @type use_l2_norm Boolean
    @type use_sqrt_norm Boolean
    @return VLAD vector of the image
    @rtype Numpy array of shape (128*K,)
    """
    vocabulary = np.load(vocabulary_path)
    vlad = np.zeros(vocabulary.shape, dtype=np.float64)
    quantizer = skln.NearestNeighbors(n_neighbors=1, algorithm="brute").fit(vocabulary)
    ws = quantizer.kneighbors(descriptors, return_distance=False).reshape(-1)

    # compute residuals
    for i in range(len(vlad)):
        if (ws == i).any():
            vlad[i, :] = np.sum(descriptors[ws == i] - vocabulary[i], axis=0)

    # square root normalization
    if use_sqrt_norm:
        vlad[:] = np.sign(vlad) * np.sqrt(np.abs(vlad))

    vlad = vlad.reshape((vlad.shape[0] * vlad.shape[1],))
    if use_l2_norm:
        vlad[:] = vlad / np.maximum(np.linalg.norm(vlad), 1e-12)

    return vlad


def vlad_pca(vlad_vectors, n_components=100):
    """Reduce VLAD vectors dimension using PCA.

    Args:
        vlad_vectors: Array of VLAD vectors of shape (N, D) where N is the number
            of images and D is the VLAD dimension (K * 128).
        n_components: Number of PCA components to keep.

    Returns:
        Tuple of (reduced_vectors, pca_model, mean_vector):
            - reduced_vectors: Array of shape (N, n_components)
            - pca_model: Fitted PCA model for transforming new vectors
            - mean_vector: Mean vector used for centering
    """
    # Center the vectors
    mean_vector = np.mean(vlad_vectors, axis=0)
    centered_vectors = vlad_vectors - mean_vector

    # Apply PCA
    pca = PCA(n_components=n_components)
    reduced_vectors = pca.fit_transform(centered_vectors)

    return reduced_vectors, pca, mean_vector


def vlad_pca_transform(vlad_vectors, pca_model, mean_vector):
    """Transform VLAD vectors using a fitted PCA model.

    Args:
        vlad_vectors: Array of VLAD vectors to transform.
        pca_model: Fitted PCA model from vlad_pca.
        mean_vector: Mean vector used for centering.

    Returns:
        Reduced VLAD vectors.
    """
    centered_vectors = vlad_vectors - mean_vector
    return pca_model.transform(centered_vectors)


def dataset_to_vlad(folder_dir):
    """Compute VLAD vector for all images in a directory.

    Args:
        folder_dir: Path to the directory containing the images.
        descriptor_func: Function that takes an image (numpy array) and returns
            a descriptor (numpy array).

    Returns:
        Dictionary mapping image filenames to their computed descriptors.
    """
    vlads = {}

    for image_path in os.listdir(folder_dir):
        image = cv2.imread(os.path.join(folder_dir, image_path))
        kp, desc = sift_kp_desc(image)
        vlads[image_path] = vlad(desc)

    return vlads


def dataset_to_vlad_pca(folder_dir, n_components=100):
    """Compute VLAD vectors and reduce dimensions using PCA for all images in a directory.

    Args:
        folder_dir: Path to the directory containing the images.
        n_components: Number of PCA components to keep.

    Returns:
        Tuple of (vlads_dict, pca_model, mean_vector):
            - vlads_dict: Dictionary mapping image filenames to their reduced VLAD vectors
            - pca_model: Fitted PCA model for transforming new vectors
            - mean_vector: Mean vector used for centering
    """
    vlads_raw = dataset_to_vlad(folder_dir)
    image_names = list(vlads_raw.keys())
    vlad_matrix = np.array([vlads_raw[name] for name in image_names])
    reduced_matrix, pca_model, mean_vector = vlad_pca(vlad_matrix, n_components)

    vlads_dict = {}
    for i, name in enumerate(image_names):
        vlads_dict[name] = reduced_matrix[i]

    return vlads_dict
