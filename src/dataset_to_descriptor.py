import os

import cv2


def dataset_to_descriptor(folder_dir: str, descriptor_func):
    """
    Compute each descriptor for each image in a directory and return a list of descriptor.
    Args:
        folder_dir (str): path to the directory containing all the images
    """
    descriptors = {}

    for image_path in os.listdir(folder_dir):
        descriptors[image_path] = descriptor_func(
            cv2.imread(folder_dir + "/" + image_path)
        )

    return descriptors
