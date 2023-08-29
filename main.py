import io
import cv2
import base64

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from model_utils import extract_bndbox_values
from utils import predict, predictv2
from PIL import Image as PILImage
import xml.etree.ElementTree as ET
from fastapi import FastAPI, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import time
from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="templates")

app = FastAPI()
app.mount('/static', StaticFiles(directory='static', html=True), name='static')

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("item.html", {"request": request})


@app.post("/prediction/")
async def get_prediction(image: UploadFile, annotations: UploadFile):
    start_time = time.time()

    # Read bites image to Pillow image
    image_obj = PILImage.open(io.BytesIO(await image.read()))

    # Read annotations
    xml_obj = ET.parse(io.BytesIO(await annotations.read()))
    print("Encoding data time: %s seconds" % (time.time() - start_time))

    # Get predicted image
    start_time = time.time()
    bnbx_values = extract_bndbox_values(xml_obj)
    result_image = predictv2(image_obj,bnbx_values )
    print("Get prediction time: %s seconds" % (time.time() - start_time))

    start_time = time.time()
    # Encode image to display it back in the interface
    _, encoded_img = cv2.imencode(".JPEG", result_image)
    # encoded_img = base64.b64encode(encoded_img)
    encoded_img_base64 = base64.b64encode(encoded_img)

    print("Decoding data time: %s seconds" % (time.time() - start_time))

    # return encoded_img_base64
    return {
        "encoded_img": encoded_img_base64,
    }


# @app.get("/prediction_from_local/",include_in_schema=False)
# async def get_prediction(filename: str | None):
#     image_dir = os.path.join("images", filename)
#     xml_file = [
#         file
#         for file in os.listdir("annotations")
#         if file.endswith(".xml") and os.path.splitext(file)[0] == filename.split(".")[0]
#     ]
#     result_image = predict(image_dir, xml_file[0], require_parsing=True)

#     return {"image": result_image}


# @app.post("/upload_annotation/")
# async def upload_image(annotation: UploadFile,include_in_schema=False):
#     # Create the "images" directory if it doesn't exist
#     os.makedirs("annotations", exist_ok=True)

#     # Save the uploaded image to the "images" directory
#     file_path = os.path.join("annotations", annotation.filename)
#     with open(file_path, "wb") as file:
#         file.write(await annotation.read())

#     return {"image": {"filename": annotation.filename}}


# @app.post("/upload_image/")
# async def upload_image(image: UploadFile,include_in_schema=False):
#     # Create the "images" directory if it doesn't exist
#     # os.makedirs("images", exist_ok=True)

#     # Save the uploaded image to the "images" directory
#     file_path = os.path.join("images", image.filename)
#     with open(file_path, "wb") as file:
#         file.write(await image.read())

#     return {"image": {"filename": image.filename}}
