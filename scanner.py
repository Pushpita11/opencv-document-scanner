import cv2
import numpy as np


def order_points(points):
    points = points.reshape(4, 2)

    ordered = np.zeros((4, 2), dtype=np.float32)

    total = points.sum(axis=1)

    ordered[0] = points[np.argmin(total)]   # Top-left
    ordered[2] = points[np.argmax(total)]   # Bottom-right

    difference = np.diff(points, axis=1)

    ordered[1] = points[np.argmin(difference)]   # Top-right
    ordered[3] = points[np.argmax(difference)]   # Bottom-left

    return ordered


def perspective_transform(image, points):

    rect = order_points(points)

    top_left = rect[0]
    top_right = rect[1]
    bottom_right = rect[2]
    bottom_left = rect[3]

    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)

    max_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bottom_left - top_left)
    height_right = np.linalg.norm(bottom_right - top_right)

    max_height = int(max(height_left, height_right))

    destination = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(
        rect,
        destination
    )

    warped = cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height)
    )

    return warped


# --------------------------------
# 1. Read image
# --------------------------------

image = cv2.imread("input/document.png")

if image is None:
    print("Image not found!")
    exit()

print("Image loaded successfully!")


# --------------------------------
# 2. Resize
# --------------------------------

image = cv2.resize(image, (800, 600))


# --------------------------------
# 3. Grayscale
# --------------------------------

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)


# --------------------------------
# 4. Blur
# --------------------------------

blur = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)


# --------------------------------
# 5. Canny Edge Detection
# --------------------------------

edges = cv2.Canny(
    blur,
    50,
    150
)


# --------------------------------
# 6. Dilate edges
# --------------------------------

kernel = np.ones((5, 5), np.uint8)

edges = cv2.dilate(
    edges,
    kernel,
    iterations=1
)


# --------------------------------
# 7. Find contours
# --------------------------------

contours, hierarchy = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print("Contours found:", len(contours))


# --------------------------------
# 8. Sort contours
# --------------------------------

contours = sorted(
    contours,
    key=cv2.contourArea,
    reverse=True
)


document = None


# --------------------------------
# 9. Find four-corner contour
# --------------------------------

for contour in contours:

    area = cv2.contourArea(contour)

    print("Contour area:", area)

    # Ignore small objects
    if area < 5000:
        continue

    perimeter = cv2.arcLength(
        contour,
        True
    )

    epsilon = 0.03 * perimeter

    approx = cv2.approxPolyDP(
        contour,
        epsilon,
        True
    )

    print("Number of corners:", len(approx))

    if len(approx) == 4:

        document = approx

        print("Document detected!")

        break


# --------------------------------
# 10. If document found
# --------------------------------

if document is not None:

    # Draw detected document
    detected = image.copy()

    cv2.drawContours(
        detected,
        [document],
        -1,
        (0, 255, 0),
        4
    )

    # Perspective transformation
    scanned = perspective_transform(
        image,
        document
    )

    # Convert to grayscale
    scanned_gray = cv2.cvtColor(
        scanned,
        cv2.COLOR_BGR2GRAY
    )

    # Improve scan
    scanned_final = cv2.adaptiveThreshold(
        scanned_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # Save
    cv2.imwrite(
        "output/scanned_document.png",
        scanned_final
    )

    print("Document scanned successfully!")
    print("Saved to output/scanned_document.png")

    # Show results
    cv2.imshow(
        "Detected Document",
        detected
    )

    cv2.imshow(
        "Scanned Document",
        scanned_final
    )

else:

    print("No document detected!")

    # Show edges so we can debug
    cv2.imshow(
        "Edges",
        edges
    )

    cv2.imshow(
        "Original",
        image
    )


cv2.waitKey(0)
cv2.destroyAllWindows()