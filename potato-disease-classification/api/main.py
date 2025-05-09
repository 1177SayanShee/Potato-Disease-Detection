# Orginal Code ----------------------------------------------------------------->
# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import tensorflow as tf

# app = FastAPI()

# def read_file_as_image(data) -> np.ndarray:
#     image = Image.open(BytesIO(data)).convert("RGB")
#     # image = image.resize((224, 224))
#     image = image.resize((256, 256))
#     return np.array(image)

# origins = [
#     "http://localhost",
#     "http://localhost:3000",
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# MODEL = tf.keras.models.load_model("../models/2")

# CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# @app.get("/ping")
# async def ping():
#     return "Hello, I am alive"

# # def read_file_as_image(data) -> np.ndarray:
# #     image = np.array(Image.open(BytesIO(data)))
# #     return image

# # @app.post("/predict")
# # async def predict(
# #     file: UploadFile = File(...)
# # ):
    
# #     image = read_file_as_image(await file.read())
# #     img_batch = np.expand_dims(image, 0)

   
    
# #     predictions = MODEL.predict(img_batch)

# #     predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
# #     confidence = np.max(predictions[0])
    
# #     return {
# #         'class': predicted_class,
# #         'confidence': float(confidence)
# #     }



# # our code ------------------------------------------------------>

# @app.post("/predict")
# async def predict(
#     file: UploadFile = File(...)
# ):
    
#     image = read_file_as_image(await file.read())
#     print(image.shape)

#     img_batch = np.expand_dims(image, 0)
#     print(img_batch.shape)  # should be (1, 224, 224, 3)

    
#     predictions = MODEL.predict(img_batch)

#     predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
#     confidence = np.max(predictions[0])
    
#     return {
#         'class': predicted_class,
#         'confidence': float(confidence)
#     }

# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port=8000)

# To this  ------------------------------------------------------------------------------->





# from fastapi import FastAPI, File, UploadFile, Request
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from fastapi.middleware.cors import CORSMiddleware
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import tensorflow as tf
# import os

# app = FastAPI()



# # CORS config (keep if using frontend separately)
# origins = ["http://localhost", "http://localhost:3000"]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load model
# MODEL = tf.keras.models.load_model("../models/2")
# CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# # Setup static files & templates
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")


# # Image read helper
# def read_file_as_image(data) -> np.ndarray:
#     image = Image.open(BytesIO(data)).convert("RGB")
#     image = image.resize((256, 256))  # match your model's input shape
#     return np.array(image)


# @app.get("/", response_class=HTMLResponse)
# async def index(request: Request):
#     return templates.TemplateResponse("index1.html", {"request": request})


# @app.post("/predict", response_class=HTMLResponse)
# async def predict(request: Request, file: UploadFile = File(...)):
#     image = read_file_as_image(await file.read())
#     img_batch = np.expand_dims(image, 0)

#     predictions = MODEL.predict(img_batch)
#     predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
#     confidence = float(np.max(predictions[0]))

#     return templates.TemplateResponse("result1.html", {
#         "request": request,
#         "predicted_class": predicted_class,
#         "confidence": f"{confidence * 100:.2f}%",
#     })



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
MODEL = tf.keras.models.load_model("../models/2")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Setup static files & templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# Image read helper
def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((256, 256))  # match your model's input shape
    return np.array(image)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
    # return templates.TemplateResponse("indexOriginal.html", {"request": request})



@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return templates.TemplateResponse("result.html", {
        "request": request,
        "predicted_class": predicted_class,
        "confidence": f"{confidence * 100:.2f}%",
    })

    # return templates.TemplateResponse("resultOriginal.html", {
    #     "request": request,
    #     "predicted_class": predicted_class,
    #     "confidence": f"{confidence * 100:.2f}%",
    # })
