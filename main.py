from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import os

app = FastAPI()

# Dynamically build the path to the model
BASE_DIR = os.path.dirname(os.path.abspath(_file_))
MODEL_PATH = os.path.join(BASE_DIR, "brain_stroke_classifier.h5")

model = load_model(MODEL_PATH)
class_labels = ["Normal", "Stroke"]

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((224, 224)).convert("RGB")
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.get("/")
def root():
    return {"message": "Brain Stroke Classification API"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = preprocess_image(image_bytes)
    prediction = model.predict(image)[0][0]
    label = class_labels[1 if prediction >= 0.5 else 0]
    confidence = float(prediction if prediction >= 0.5 else 1 - prediction)
    return {
        "prediction": label,
        "confidence": round(confidence, 4)
    }

