import cv2
import numpy as np
import tensorflow as tf

# =========================
# Paths
# =========================

MODEL_PATH = "models/face_mask_mobilenetv2.keras"
FACE_PROTO = "models/deploy.prototxt"
FACE_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"

# =========================
# Load Models
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
# Start Camera
# =========================

print("Starting camera...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

print("Camera started!")
print("Press 'q' to quit.")

# =========================
# Main Loop
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    (h, w) = frame.shape[:2]

    # Create blob for face detector
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    face_net.setInput(blob)
    detections = face_net.forward()

    # Detect faces
    for i in range(detections.shape[2]):

        confidence = float(detections[0, 0, i, 2])

        if confidence < 0.50:
            continue

        # Get face coordinates
        box = detections[0, 0, i, 3:7] * np.array(
            [w, h, w, h]
        )

        x1, y1, x2, y2 = box.astype(int)

        # Keep coordinates inside image
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # Extract face
        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        # Convert BGR -> RGB
        face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        # Resize for MobileNetV2
        face = cv2.resize(
            face,
            (224, 224)
        )

        # Convert to float
        face = face.astype("float32")

        # Add batch dimension
        face = np.expand_dims(
            face,
            axis=0
        )

        # =========================
        # Mask Prediction
        # =========================

        prediction = model.predict(
            face,
            verbose=0
        )[0][0]

        if prediction >= 0.5:

            label = "Without Mask"
            mask_confidence = prediction

        else:

            label = "With Mask"
            mask_confidence = 1 - prediction

        # =========================
        # Draw Result
        # =========================

        text = f"{label}: {mask_confidence:.1%}"

        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Draw label background
        cv2.rectangle(
            frame,
            (x1, y1 - 35),
            (x2, y1),
            (0, 255, 0),
            -1
        )

        # Draw text
        cv2.putText(
            frame,
            text,
            (x1 + 5, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    # Show camera
    cv2.imshow(
        "Face Mask Detection",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================
# Cleanup
# =========================

cap.release()
cv2.destroyAllWindows()

print("Camera stopped.")