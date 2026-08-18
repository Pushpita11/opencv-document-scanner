import streamlit as st
import cv2
import numpy as np


st.set_page_config(
    page_title="OpenCV Document Scanner",
    page_icon="📄",
    layout="wide"
)

st.title("📄 OpenCV Document Scanner")
st.write("Upload a document image and convert it into a scanned document.")


uploaded_file = st.file_uploader(
    "Upload your document",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Read uploaded image
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image is not None:

        # Resize
        image = cv2.resize(image, (800, 600))

        # Grayscale
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # Blur
        blur = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        # Edge detection
        edges = cv2.Canny(
            blur,
            75,
            200
        )

        # Find contours
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        contours = sorted(
            contours,
            key=cv2.contourArea,
            reverse=True
        )

        document = None

        # Find document
        for contour in contours:

            area = cv2.contourArea(contour)

            if area < 5000:
                continue

            perimeter = cv2.arcLength(
                contour,
                True
            )

            approx = cv2.approxPolyDP(
                contour,
                0.03 * perimeter,
                True
            )

            if len(approx) == 4:
                document = approx
                break

        if document is not None:

            st.success("Document detected successfully!")

            # Draw detected document
            detected = image.copy()

            cv2.drawContours(
                detected,
                [document],
                -1,
                (0, 255, 0),
                4
            )

            # Show images
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                st.image(
                    cv2.cvtColor(
                        image,
                        cv2.COLOR_BGR2RGB
                    ),
                    use_container_width=True
                )

            with col2:
                st.subheader("Detected Document")
                st.image(
                    cv2.cvtColor(
                        detected,
                        cv2.COLOR_BGR2RGB
                    ),
                    use_container_width=True
                )

            st.info(
                "Document detected. Your OpenCV scanner "
                "successfully identified the document boundary."
            )

        else:

            st.error(
                "No document detected. "
                "Please upload a clear photo of a full document."
            )

    else:

        st.error("Unable to read the uploaded image.")