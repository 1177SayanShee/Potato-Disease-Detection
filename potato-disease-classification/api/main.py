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
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
from fastapi import BackgroundTasks
import time
load_dotenv()  

# Configure using your Cloudinary credentials
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

# --- Get base directory of this script (main.py) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


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
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "3"))



MODEL = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# Image read helper
def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((256, 256))  # match our model's input shape
    return np.array(image)

# background task to delete the image
def delete_image_after_delay(public_id: str, delay: int = 300):
    time.sleep(delay)  # wait for 5 minutes
    cloudinary.uploader.destroy(public_id, invalidate=True)


# Route Handlers ----------------------------------------------------->
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    background_tasks: BackgroundTasks,  
    file: UploadFile = File(None),
    camera_image: str = Form(None)
):
    image = None
    image_url = None
    public_id = None  

    # File uploaded manually
    if file is not None:
        contents = await file.read()
        image = read_file_as_image(contents)

        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(
           contents,
           folder="potato_disease_uploads/",
           transformation=[
           {"width": 256, "height": 256, "crop": "limit"},
           {"fetch_format": "auto", "quality": "auto"}
            ],
           invalidate=True  # Important for deletion
        )

        image_url = upload_result.get("secure_url")
        public_id = upload_result.get("public_id")  # Capture public_id


    # Camera image as base64 string
    elif camera_image and camera_image.strip() != "":

        header, encoded = camera_image.split(",", 1)
        decoded = base64.b64decode(encoded)
        image = read_file_as_image(decoded)

        upload_result = cloudinary.uploader.upload(
           decoded,
           folder="potato_disease_uploads/",
           transformation=[
           {"width": 256, "height": 256, "crop": "limit"},
           {"fetch_format": "auto", "quality": "auto"}
            ],
           invalidate=True
        )


        image_url = upload_result.get("secure_url")
        public_id = upload_result.get("public_id")

    else:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "No image provided"
        })
    
    # Schedule the deletion task
    if public_id:
        background_tasks.add_task(delete_image_after_delay, public_id, delay=300) # 300 seconds
        
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

