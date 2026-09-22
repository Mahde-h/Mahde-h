from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import cv2
import numpy as np
import tensorflow as tf


app = FastAPI()


# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Load Face Detector
# =========================

FACE_PROTO = "models/deploy.prototxt"
FACE_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"

face_net = cv2.dnn.readNetFromCaffe(
    FACE_PROTO,
    FACE_MODEL
)


# =========================
# Load Mask Model
# =========================

MASK_MODEL = "models/face_mask_mobilenetv2.keras"

mask_model = tf.keras.models.load_model(
    MASK_MODEL,
    compile=False
)


# =========================
# API Test
# =========================

@app.get("/")
def root():
    return {
        "status": "running",
        "message": "Face Mask Detection API is working"
    }


# =========================
# Prediction
# =========================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded image
    contents = await file.read()

    # Convert bytes to NumPy array
    np_array = np.frombuffer(contents, np.uint8)

    # Decode image
    frame = cv2.imdecode(
        np_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        return {
            "error": "Could not read image"
        }

    h, w = frame.shape[:2]

    # =========================
    # Face Detection
    # =========================

    blob = cv2.dnn.blobFromImage(
        frame,
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    face_net.setInput(blob)

    detections = face_net.forward()

    results = []

    # =========================
    # Process Faces
    # =========================

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence < 0.5:
            continue

        box = detections[0, 0, i, 3:7] * np.array(
            [w, h, w, h]
        )

        x1, y1, x2, y2 = box.astype(int)

        # Keep coordinates inside image
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        # =========================
        # MobileNetV2 Preprocessing
        # =========================

        face_resized = cv2.resize(
            face,
            (224, 224)
        )

        face_rgb = cv2.cvtColor(
            face_resized,
            cv2.COLOR_BGR2RGB
        )

        face_input = face_rgb.astype(
            np.float32
        ) 

        face_input = np.expand_dims(
            face_input,
            axis=0
        )

        # =========================
        # Mask Prediction
        # =========================

        prediction = mask_model.predict(
            face_input,
            verbose=0
        )[0][0]

        # IMPORTANT:
        # 0 = With Mask
        # 1 = Without Mask
        #
        # حسب تدريب موديلك السابق

        if prediction >= 0.5:
            label = "Without Mask"
            mask_probability = prediction
        else:
            label = "With Mask"
            mask_probability = 1 - prediction

        results.append({
            "face_confidence": float(confidence),
            "mask_prediction": float(prediction),
            "label": label,
            "mask_probability": float(mask_probability),
            "box": {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2)
            }
        })

    return {
        "faces": len(results),
        "results": results
    }