
import cv2

from dataset_to_descriptor import gray_scale_dataset


def sift_kp_desc(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, desc = sift.detectAndCompute(gray_image, None)
    return kp, desc


def show_keypoints(image_path, keypoints):
    image = cv2.imread(image_path)

    image_with_keypoints = cv2.drawKeypoints(
        image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    cv2.imshow("SIFT Keypoints", image_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
