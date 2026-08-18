import cv2

# Read image
image = cv2.imread("input/document.png")

if image is None:
    print("Image not found!")
    exit()

# Resize image
image = cv2.resize(image, (800, 600))

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur image
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect edges
edges = cv2.Canny(blur, 75, 200)

# Find contours
contours, hierarchy = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Sort contours by area
contours = sorted(
    contours,
    key=cv2.contourArea,
    reverse=True
)

# Check contours
for contour in contours:

    area = cv2.contourArea(contour)

    # Ignore very small objects
    if area < 1000:
        continue

    # Approximate contour
    perimeter = cv2.arcLength(contour, True)

    epsilon = 0.02 * perimeter

    approx = cv2.approxPolyDP(
        contour,
        epsilon,
        True
    )

    # Check if contour has 4 corners
    if len(approx) == 4:

        print("Document detected!")
        print("Area:", area)

        # Draw the detected document
        cv2.drawContours(
            image,
            [approx],
            -1,
            (0, 255, 0),
            3
        )

        break

# Show result
cv2.imshow("Detected Document", image)
cv2.imshow("Edges", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()