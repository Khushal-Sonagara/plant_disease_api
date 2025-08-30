import os
import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps
import uvicorn

# -----------------------
# Config
# -----------------------
MODEL_PATH = os.getenv("MODEL_PATH", "trainedd_model_fixed.keras")
IMG_SIZE = 128  # must match training

# -----------------------
# Load Model
# -----------------------
model = tf.keras.models.load_model(MODEL_PATH)

# Class names (must match training order!)
class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# -----------------------
# FastAPI App
# -----------------------
app = FastAPI(title="Plant Disease Detection API")

@app.get("/")
async def root():
    return {"status": "ok", "message": "Plant Disease Detection API is running"}

def preprocess_image(image_bytes: bytes):
    """Resize and normalize image"""
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.asarray(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        input_arr = preprocess_image(image_bytes)

        preds = model.predict(input_arr)
        result_index = np.argmax(preds)
        confidence = float(np.max(preds))

        return JSONResponse(content={
            "disease": class_names[result_index],
            "confidence": round(confidence, 4)
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    # For local testing
    uvicorn.run(app, host="0.0.0.0", port=8000)
