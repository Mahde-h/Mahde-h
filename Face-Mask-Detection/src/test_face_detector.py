import cv2
import numpy as np

PROTO = "models/deploy.prototxt"
MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"
IMAGE_PATH = "test_images/WIN_20260914_15_21_26_Pro.jpg"

print("Loading face detector...")

net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)

print("Face detector loaded!")

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError("Image could not be loaded")

print("Image shape:", image.shape)

(h, w) = image.shape[:2]

blob = cv2.dnn.blobFromImage(
    cv2.resize(image, (300, 300)),
    1.0,
    (300, 300),
    (104.0, 177.0, 123.0)
)

net.setInput(blob)

detections = net.forward()

print("Detection shape:", detections.shape)

found = 0

for i in range(detections.shape[2]):

    confidence = float(detections[0, 0, i, 2])

    print(f"Detection {i}: confidence = {confidence:.4f}")

    if confidence > 0.20:
        found += 1

print("\nFaces detected:", found)

max_confidence = 0.0
max_index = -1

for i in range(detections.shape[2]):
    confidence = float(detections[0, 0, i, 2])

    if confidence > max_confidence:
        max_confidence = confidence
        max_index = i

print("\n==============================")
print("Highest confidence:", max_confidence)
print("Detection index:", max_index)
print("==============================")