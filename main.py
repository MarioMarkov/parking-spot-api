import io
import os
import cv2
import time
import base64
from PIL import Image as PILImage

from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from utils.inference_utils import predict
from utils.model_utils import extract_bndbox_values


from dotenv import load_dotenv

from video_predict import gen_streaming_frames, gen_video_chunks

# Load the .env file
load_dotenv()


templates = Jinja2Templates(directory="templates")

app = FastAPI()
#app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/static/example", StaticFiles(directory="example"), name="example")


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

    return templates.TemplateResponse(
        "item.html", {"request": request, "env_var": os.environ["DEPLOYMENT_URL"]}
    )


@app.post("/prediction")
async def get_prediction(image: UploadFile, annotations: UploadFile):
    start_time = time.time()

    # Read bites image to Pillow image
    image_obj = PILImage.open(io.BytesIO(await image.read()))
    # Read annotations
    print("Encoding data time: %s seconds" % (time.time() - start_time))

    # Get predicted image
    start_time = time.time()

    annotation_file = io.BytesIO(await annotations.read())

    bnbx_values = extract_bndbox_values(annotation_file)

    result_image = predict(
        image_obj, bnbx_values, video=False, batch_size=8, threshhold=0.6
    )

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


@app.get("/video")
def get_video():
    file_path = "./example/parking_video.mp4"  # replace with your video file path

    # Open video file in binary mode
    def iterfile():
        with open(file_path, mode="rb") as file_like:
            yield from file_like

    # Return a streaming response
    return StreamingResponse(iterfile(), media_type="video/mp4")


@app.get("/video")
def get_video():
    file_path = "video_chunks\output_chunk.mp4"

    # Return a streaming response
    return FileResponse(file_path, media_type="video/mp4")


@app.get("/video_chunk/{start_second}/parking_video.mp4")
async def get_video_chunk(start_second: int):
    video_path = f"video_chunks\output_chunk_st_{start_second}_end_{start_second+5}.mp4"

    if not os.path.exists(video_path):
        print("video chunk does not exist, generating...")
        annotation_path = "./example/pg_survailance.xml"
        print(f"Starting from second {start_second}")
        bndbox_values = extract_bndbox_values(annotation_path)

        gen_video_chunks(
            "./example/parking_video.mp4",
            step=60,
            bndbox_values=bndbox_values,
            start_second=start_second,
        )

    # Check if the file exists
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")

    # Serve the video file
    return FileResponse(video_path, media_type="video/mp4")


@app.get("/video_pred")
def get_video():
    video_path = "./example/parking_video.mp4"
    annotation_path = "./example/pg_survailance.xml"

    bndbox_values = extract_bndbox_values(annotation_path)

    # Return a streaming response
    return StreamingResponse(
        gen_streaming_frames(video_path, step=60, bndbox_values=bndbox_values),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/video-stream.m3u8")
def video_stream():
    return FileResponse("stream/stream.m3u8")


@app.get("/segments/{segment_name}")
def video_segments(segment_name: str):
    return FileResponse(f"stream/{segment_name}")


if __name__ == "__main__":
    import uvicorn

    if os.environ["DEPLOYMENT_URL"] == "http://localhost:8000":
        import ngrok

        port = 8000
        listener = ngrok.forward(
            f"http://localhost:{port}",
            authtoken_from_env=True,
            domain="possum-enough-informally.ngrok-free.app",
        )
        public_url = listener.url()
        print(f"Waiting on public url: {public_url}")

    uvicorn.run("main:app", host="localhost", port=port, reload=False)
