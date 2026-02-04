import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def color_histogram(im, bins_per_channel=8):
    """Compute a normalized joint color histogram.

    Args:
        im: Color image as numpy array of shape (height, width, 3) and dtype uint8.
        bins_per_channel: Number of bins per channel after quantization.

    Returns:
        Normalized color histogram as numpy array of shape (bins_per_channel**3,)
        and dtype float32.
    """
    im = im.copy()

    # Quantize image
    bin_width = 256.0 / bins_per_channel
    im = (im / bin_width).astype(np.uint32)

    # Flatten color space
    im = im[..., 0] * bins_per_channel**2 + im[..., 1] * bins_per_channel + im[..., 2]

    # Compute and normalize histogram
    histogram = np.zeros((bins_per_channel**3,), dtype=np.float32)
    colors, counts = np.unique(im, return_counts=True)
    histogram[colors] = counts
    histogram = histogram / np.linalg.norm(histogram, ord=1)

    return histogram


def lbp_histogram(im, n_points=8, radius=1):
    """Compute Local Binary Pattern histogram of an image.

    Args:
        im: Color image as numpy array of shape (height, width, 3) and dtype uint8.
        n_points: Number of circularly symmetric neighbor points.
        radius: Radius of circle for neighbor points.

    Returns:
        Normalized LBP histogram as numpy array.
    """
    gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    lbp_image = local_binary_pattern(gray_image, n_points, radius)
    n_bins = int(lbp_image.max() + 1)
    hist, _ = np.histogram(lbp_image, bins=np.arange(n_bins + 1), density=True)

    return hist


def fusion_histogram(im):
    """Compute concatenated color and LBP histograms.

    Combines global color information with local texture patterns
    for a more discriminative descriptor.

    Args:
        im: Color image as numpy array of shape (height, width, 3) and dtype uint8.

    Returns:
        Concatenated histogram as numpy array combining color (512 dims)
        and LBP histograms.
    """
    color_h = color_histogram(im)
    lbp_h = lbp_histogram(im)

    return np.hstack((color_h, lbp_h))


def local_desc(im, grid_size=5):
    """Compute spatial pyramid descriptor using color histograms.

    Divides the image into a grid and computes a color histogram for each cell,
    preserving spatial information about color distribution.

    Args:
        im: Color image as numpy array of shape (height, width, 3) and dtype uint8.
        grid_size: Number of cells per row/column (default 5 for a 5x5 grid).

    Returns:
        Concatenated histograms from all cells as numpy array of shape
        (grid_size**2 * bins_per_channel**3,).
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
