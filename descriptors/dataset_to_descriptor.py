
import os

import cv2


def gray_scale_dataset(folder_dir):
    images = {}

    for image_path in os.listdir(folder_dir):
        image = cv2.imread(os.path.join(folder_dir, image_path))
        images[image_path] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return images


def dataset_to_descriptor(folder_dir, descriptor_func):
    descriptors = {}

    for image_path in os.listdir(folder_dir):
        image = cv2.imread(os.path.join(folder_dir, image_path))
        descriptors[image_path] = descriptor_func(image)

    return descriptors
