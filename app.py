import os
import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
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

# Sanity check: class count vs model outputs
try:
    n_out = model.output_shape[-1]
    if n_out != len(class_names):
        print(f"WARNING: model outputs {n_out} but you supplied {len(class_names)} class names.")
except Exception:
    # model.output_shape might be complicated for some custom models; ignore if not available
    pass

# Warm-up (avoid first-call latency)
try:
    _ = model.predict(np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32))
except Exception:
    pass

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title="Plant Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok", "message": "Plant Disease Detection API is running"}

def preprocess_using_user_code(image_bytes: bytes):
    """
    Uses the exact sequence you used locally:
      - tf.keras.preprocessing.image.load_img from bytes
      - tf.keras.preprocessing.image.img_to_array
      - wrap into a batch (no /255 normalization)
    """
    # load_img accepts a file-like (BytesIO), and target_size
    pil_img = tf.keras.preprocessing.image.load_img(io.BytesIO(image_bytes), target_size=(IMG_SIZE, IMG_SIZE))
    arr = tf.keras.preprocessing.image.img_to_array(pil_img)          # shape (H, W, C), dtype float32
    batch = np.array([arr])                                           # shape (1, H, W, C)
    return batch

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        input_arr = preprocess_using_user_code(image_bytes)

        preds = model.predict(input_arr)

        # Make preds a 1D vector of probabilities/logits
        preds = np.squeeze(preds)
        if preds.ndim == 0:
            # single scalar? unexpected
            return JSONResponse(content={"error": "Unexpected model output shape"}, status_code=500)

        result_index = int(np.argmax(preds))
        confidence = float(np.max(preds))

        # safe bounds
        if result_index < 0 or result_index >= len(class_names):
            return JSONResponse(content={"error": "Predicted index outside class list bounds"}, status_code=500)

        return JSONResponse(content={
            "disease": class_names[result_index],
            "confidence": round(confidence, 6)
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8000))  # default 8000 for local testing
    uvicorn.run(app, host="0.0.0.0", port=port)

