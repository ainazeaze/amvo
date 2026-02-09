import os

import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors

from sift import sift_kp_desc


def hist_word(image, vocab_path="vocabularies_sift/vocabulary_5000.npy"):
    vocab = np.load(vocab_path)
    vocab_size = vocab.shape[0]

    keypoints, descriptors = sift_kp_desc(image)

    if descriptors is None:
        return np.zeros(vocab_size)

    nn = NearestNeighbors(n_neighbors=1, algorithm="brute")
    nn.fit(vocab)
    indices = nn.kneighbors(descriptors, return_distance=False)

    histogram = np.zeros(vocab_size)
    for idx in indices:
        histogram[idx[0]] += 1

    if np.sum(histogram) > 0:
        histogram = histogram / np.sum(histogram)

    return histogram


def dataset_to_hist_word(folder_dir):
    descriptors = {}

    for image_path in os.listdir(folder_dir):
        image = cv2.imread(os.path.join(folder_dir, image_path))
        descriptors[image_path] = hist_word(image)

    return descriptors
