import cv2
import numpy as np


def order_points(points):
    """
    Arrange four points in this order:
    top-left, top-right, bottom-right, bottom-left
    """

    points = points.reshape(4, 2)

    ordered = np.zeros((4, 2), dtype=np.float32)

    # Top-left has the smallest x + y
    # Bottom-right has the largest x + y
    total = points.sum(axis=1)

    ordered[0] = points[np.argmin(total)]
    ordered[2] = points[np.argmax(total)]

    # Top-right has the smallest x - y
    # Bottom-left has the largest x - y
    difference = np.diff(points, axis=1)

    ordered[1] = points[np.argmin(difference)]
    ordered[3] = points[np.argmax(difference)]

    return ordered


def perspective_transform(image, points):

    # Arrange the four corners
    rect = order_points(points)

    top_left = rect[0]
    top_right = rect[1]
    bottom_right = rect[2]
    bottom_left = rect[3]

    # Calculate width
    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)

    max_width = int(max(width_top, width_bottom))

    # Calculate height
    height_left = np.linalg.norm(bottom_left - top_left)
    height_right = np.linalg.norm(bottom_right - top_right)

    max_height = int(max(height_left, height_right))

    # Destination points
    destination = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    # Get transformation matrix
    matrix = cv2.getPerspectiveTransform(
        rect,
        destination
    )

    # Apply transformation
    warped = cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height)
    )

    return warped