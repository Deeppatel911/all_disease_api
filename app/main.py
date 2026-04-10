from fastapi import FastAPI, UploadFile, File
import numpy as np
import tensorflow as tf
from app.utils.preprocessing import preprocess_image

app = FastAPI()

# Load model once at startup
MODEL_PATH = "app/model/DenseNet121"
model = tf.keras.models.load_model(MODEL_PATH)

# Label mapping
class_names = {0: 'benign', 1: 'early', 2: 'pre', 3: 'pro'}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded image
    image_bytes = await file.read()

    # Preprocess image
    img = preprocess_image(image_bytes)

    # Run prediction
    preds = model.predict(img)
    pred_class = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))

    #print("Raw preds:", preds)
    #print("Argmax:", pred_class)
    #print("Confidence:", confidence)


    return {
        "prediction": class_names[pred_class],
        "confidence": round(confidence, 4)
    }
