import io
import cv2
import base64

from utils import predict
from PIL import Image as PILImage
import xml.etree.ElementTree as ET
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

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


@app.get("/")
async def home():
    return {"content": "Hello from parking api"}


@app.post("/prediction/")
async def get_prediction(image: UploadFile, annotations: UploadFile):
    image_obj = PILImage.open(io.BytesIO(await image.read()))
    xml_obj = ET.parse(io.BytesIO(await annotations.read()))

    result_image = predict(image_obj, xml_obj, require_parsing=False)

    # line that fixed it
    _, encoded_img = cv2.imencode(".PNG", result_image)

    encoded_img = base64.b64encode(encoded_img)

    return {
        "filename": image.filename,
        "encoded_img": encoded_img,
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
