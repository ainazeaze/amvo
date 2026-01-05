import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def color_histogram(im, bins_per_channel=8):
    """Computes a joint color histogram.
    :param im Color image as a Numpy array of shape (height, width, 3)
    @param bins_per_channel Number of bins per channel after quantization
    @type im Numpy array of type uint8 and shape (height, width, 3)
    @type bins_per_channel Integer
    @return Normalized color histogram
    @rtype Numpy array of type float32 and shape (bins_per_channel**3,)
    """
    im = im.copy()

    # quantize image
    bin_width = 256.0 / bins_per_channel
    im = (im / bin_width).astype(np.uint32)

    # flatten color space
    im = im[..., 0] * bins_per_channel**2 + im[..., 1] * bins_per_channel + im[..., 2]

    # compute and normalize histogram
    histogram = np.zeros((bins_per_channel**3,), dtype=np.float32)
    colors, counts = np.unique(im, return_counts=True)
    histogram[colors] = counts
    histogram = histogram / np.linalg.norm(histogram, ord=1)
    return histogram


def lbp_histogram(im):
    """
    Compute LBP histogram of an image
    @type im Numpy array of type uint8 and shape (height, width, 3)
    """
    gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    lbp_image = local_binary_pattern(gray_image, 8, 1)
    n_bins = int(lbp_image.max() + 1)
    hist, _ = np.histogram(lbp_image, bins=np.arange(n_bins + 1), density=True)

    return hist


def fusion_histrogram(im):
    """
    Compute and stack joint color hist and LBP hist
    @type im Numpy array of type uint8 and shape (height, width, 3)
    """
    color_h = color_histogram(im)
    print(color_h.shape)
    lbp_h = lbp_histogram(im)
    print(lbp_h.shape)

    return np.hstack((color_h, lbp_h))


def local_desc(im, grid_size=5):
    """
    Divide image into a grid and compute color histogram for each cell.
    @param im: Color image as numpy array of shape (height, width, 3)
    @param grid_size: Number of cells per row/column (default 5x5 grid)
    @return: Concatenated histograms from all cells
    """
    h, w, _ = im.shape
    cell_h = h // grid_size
    cell_w = w // grid_size

    sub_desc = []
    for i in range(grid_size):
        for j in range(grid_size):
            cell = im[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w]
            sub_desc.append(color_histogram(cell))

    return np.hstack(sub_desc)
