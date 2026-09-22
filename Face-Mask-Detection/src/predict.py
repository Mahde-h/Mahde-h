import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import os




# =========================
# Paths
# =========================

MODEL_PATH = "models/face_mask_mobilenetv2.keras"

FACE_PROTO = "models/deploy.prototxt"

FACE_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"


# =========================
# Check image path
# =========================


image_path = "test_images/WIN_20260914_15_21_26_Pro.jpg"

if not os.path.exists(image_path):
    raise FileNotFoundError(
        f"Image not found: {image_path}"
    )


# =========================
# Load models
# =========================

print("Loading mask model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Mask model loaded successfully!")


print("Loading face detector...")

face_net = cv2.dnn.readNetFromCaffe(
    FACE_PROTO,
    FACE_MODEL
)

print("Face detector loaded successfully!")


# =========================
# Load image
# =========================

image = cv2.imread(image_path)

print("Image path:", image_path)
print("Image shape:", None if image is None else image.shape)

if image is None:
    raise FileNotFoundError(
        f"Could not read image: {image_path}"
    )

(h, w) = image.shape[:2]


# =========================
# Face detection
# =========================

blob = cv2.dnn.blobFromImage(
    cv2.resize(image, (300, 300)),
    1.0,
    (300, 300),
    (104.0, 177.0, 123.0)
)

face_net.setInput(blob)

detections = face_net.forward()


boxes = []
confidences = []


for i in range(detections.shape[2]):

    confidence = float(
        detections[0, 0, i, 2]
    )

    if confidence >= 0.40:

        box = (
            detections[0, 0, i, 3:7]
            * np.array([w, h, w, h])
        )

        x1, y1, x2, y2 = box.astype(int)

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        box_w = x2 - x1
        box_h = y2 - y1

        if box_w < 20 or box_h < 20:
            continue

        boxes.append(
            [x1, y1, box_w, box_h]
        )

        confidences.append(confidence)


print("Faces before NMS:", len(boxes))


# =========================
# Non-Maximum Suppression
# =========================

indices = cv2.dnn.NMSBoxes(
    boxes,
    confidences,
    score_threshold=0.40,
    nms_threshold=0.40
)


if len(indices) == 0:

    print("No faces detected.")

else:

    print(
        "Faces after NMS:",
        len(indices)
    )


# =========================
# Convert image to RGB
# =========================

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# =========================
# Mask classification
# =========================

if len(indices) > 0:

    for face_number, idx in enumerate(
        indices.flatten(),
        start=1
    ):

        x, y, box_w, box_h = boxes[idx]

        x2 = x + box_w
        y2 = y + box_h

        face_confidence = confidences[idx]


        # -------------------------
        # Crop face
        # -------------------------

        face = image[
            y:y2,
            x:x2
        ]

        if face.size == 0:
            continue


        # BGR -> RGB

        face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )


        # Resize

        face = cv2.resize(
            face,
            (224, 224)
        )


        # Convert to float32

        face = face.astype(
            "float32"
        )


        # Add batch dimension

        face = np.expand_dims(
            face,
            axis=0
        )


        # -------------------------
        # Prediction
        # -------------------------

        prediction = model.predict(
            face,
            verbose=0
        )[0][0]


        # -------------------------
        # Classification
        # -------------------------

        if prediction >= 0.5:

            label = "Without Mask"

            mask_confidence = prediction

        else:

            label = "With Mask"

            mask_confidence = 1 - prediction


        # -------------------------
        # Print results
        # -------------------------

        print(
            f"\nFace {face_number}"
        )

        print(
            f"Face confidence: "
            f"{face_confidence:.4f}"
        )

        print(
            f"Mask prediction: "
            f"{prediction:.4f}"
        )

        print(
            f"Result: {label} "
            f"({mask_confidence:.2%})"
        )


        # -------------------------
        # Draw rectangle
        # -------------------------

        cv2.rectangle(
            image_rgb,
            (x, y),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # -------------------------
        # Draw label
        # -------------------------

        text = (
            f"{label}: "
            f"{mask_confidence:.2f}"
        )

        cv2.putText(
            image_rgb,
            text,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


# =========================
# Display result
# =========================

plt.figure(
    figsize=(10, 8)
)

plt.imshow(
    image_rgb
)

plt.axis("off")

plt.title(
    "Face Mask Detection"
)

plt.show()