import tensorflow as tf

MODEL_PATH = "models/face_mask_mobilenetv2.keras"

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")
print("\nModel Summary:")
model.summary()