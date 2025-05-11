from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi import Form
import base64

# --- Get base directory of this script (main.py) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# print("Base path being used:", BASE_DIR)

app = FastAPI()

# CORS config (keep if using frontend separately)
origins = ["http://localhost", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "2"))
# MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "saved_models", "1"))

print("Model path being used:", MODEL_PATH)

MODEL = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# Image read helper
def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((256, 256))  # match your model's input shape
    return np.array(image)



# Route Handlers ----------------------------------------------------->
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    file: UploadFile = File(None),
    camera_image: str = Form(None)
):
    image = None
    image_url = None

    # Ensure upload directory exists
    upload_dir = os.path.join(STATIC_DIR, "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    # Case 1: File uploaded manually
    if file is not None:
        contents = await file.read()
        image = read_file_as_image(contents)

        # Save the uploaded image to static/uploads
        image_filename = file.filename
        image_path = os.path.join(upload_dir, image_filename)

        with open(image_path, "wb") as f:
            f.write(contents)

        image_url = f"/static/uploads/{image_filename}"  # for browser

    # Case 2: Camera image as base64 string

    
    elif camera_image and camera_image.strip() != "":

        header, encoded = camera_image.split(",", 1)
        decoded = base64.b64decode(encoded)
        image = read_file_as_image(decoded)

        # Save to static/uploads with a default name (e.g., from webcam)
        image_filename = "webcam_image.jpg"
        image_path = os.path.join(upload_dir, image_filename)

        with open(image_path, "wb") as f:
            f.write(decoded)

        image_url = f"/static/uploads/{image_filename}"  # for browser

    else:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "No image provided"
        })

    # Run prediction
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return templates.TemplateResponse("result.html", {
        "request": request,
        "predicted_class": predicted_class,
        "confidence": f"{confidence * 100:.2f}",
        "image_url": image_url
    })


@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/solution/early-blight", response_class=HTMLResponse)
async def early_blight_solution(request: Request):
    return templates.TemplateResponse("early_blight.html", {"request": request})


@app.get("/solution/late-blight", response_class=HTMLResponse)
async def late_blight_solution(request: Request):
    return templates.TemplateResponse("late_blight.html", {"request": request})

