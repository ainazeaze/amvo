import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def color_histogram(im, bins_per_channel=8):
    im = im.copy()

    bin_width = 256.0 / bins_per_channel
    im = (im / bin_width).astype(np.uint32)

    im = im[..., 0] * bins_per_channel**2 + im[..., 1] * bins_per_channel + im[..., 2]

    histogram = np.zeros((bins_per_channel**3,), dtype=np.float32)
    colors, counts = np.unique(im, return_counts=True)
    histogram[colors] = counts
    histogram = histogram / np.linalg.norm(histogram, ord=1)

    return histogram


def lbp_histogram(im, n_points=8, radius=1):
    gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    lbp_image = local_binary_pattern(gray_image, n_points, radius)
    n_bins = int(lbp_image.max() + 1)
    hist, _ = np.histogram(lbp_image, bins=np.arange(n_bins + 1), density=True)

    return hist


def fusion_histogram(im):
    color_h = color_histogram(im)
    lbp_h = lbp_histogram(im)

    return np.hstack((color_h, lbp_h))


def local_desc(im, grid_size=5):
    h, w, _ = im.shape
    cell_h = h // grid_size
    cell_w = w // grid_size

    sub_desc = []
    for i in range(grid_size):
        for j in range(grid_size):
            cell = im[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w]
            sub_desc.append(color_histogram(cell))

    return np.hstack(sub_desc)
