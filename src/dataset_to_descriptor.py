"""Module for computing image descriptors on a dataset."""

import os

import cv2


def dataset_to_descriptor(folder_dir, descriptor_func):
    """Compute descriptors for all images in a directory.

    Args:
        folder_dir: Path to the directory containing the images.
        descriptor_func: Function that takes an image (numpy array) and returns
            a descriptor (numpy array).

    Returns:
        Dictionary mapping image filenames to their computed descriptors.
    """
    descriptors = {}

    for image_path in os.listdir(folder_dir):
        image = cv2.imread(os.path.join(folder_dir, image_path))
        descriptors[image_path] = descriptor_func(image)

    return descriptors
