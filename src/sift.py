"""Module for SIFT keypoint detection and description."""

import cv2

from dataset_to_descriptor import gray_scale_dataset


def sift_kp_desc(image):
    """Compute SIFT keypoints and descriptors for an image

    Args:
        image_path : path to the image

    Returns:
        Tuple of two lists:
            - keypoints: List of keypoints for each image
            - descriptors: List of descriptors for each image
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, desc = sift.detectAndCompute(gray_image, None)
    return kp, desc


def show_keypoints(image_path, keypoints):
    """Display an image with its detected SIFT keypoints.

    Args:
        image_path: Path to the image file.
        keypoints: List of cv2.KeyPoint objects detected in the image.
    """
    image = cv2.imread(image_path)

    image_with_keypoints = cv2.drawKeypoints(
        image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    cv2.imshow("SIFT Keypoints", image_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
