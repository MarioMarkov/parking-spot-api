import cv2
import torch
import numpy as np
from PIL import Image

from utils.inference_utils import draw_rectangles_on_batches, predict
from utils.model_utils import (
    extract_bndbox_values,
    mAlexNet,
)


# device = "cpu"
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")

# print(f"Using {device}")

# video_path = "./example/parking_video.mp4"
# annotation_path = "./example/pg_survailance.xml"
# model_path = "./models/m_alex_net_both_best_acc.pth"


# model = mAlexNet().to(device)

# model.load_state_dict(
#     torch.load(model_path, map_location=torch.device(device), weights_only=True)
# )
# model.eval()
# bndbox_values = extract_bndbox_values(annotation_path)


## Inference video

# cap = cv2.VideoCapture(video_path)
# ret = True
# frame_num = 0
# step = 60
# spots = {}


# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     full_image = Image.fromarray(np.uint8(frame))

#     # Every step frame refresh values
#     if frame_num % step == 0:
#         predicted_spots = predict(
#             full_image,
#             bndbox_values=bndbox_values,
#             video=True,
#             batch_size=16,
#             threshhold=0.6,
#         )
#         spots.update(predicted_spots)

#     # this frame is the image, for loop over all the spots
#     draw_rectangles_on_batches(image=frame, bndbox_values=bndbox_values, spots=spots)
#     frame_num += 1

#     cv2.imshow("Parking Lot", frame)
#     # Break the loop if 'q' key is pressed
#     if cv2.waitKey(25) & 0xFF == ord("q"):
#         break


# cap.release()
# cv2.destroyAllWindows()


def gen_video_chunks(
    model, video_path: str, step: int, bndbox_values: dict, start_second
):

    cap = cv2.VideoCapture(video_path)
    ret = True
    frame_num = 0
    spots = {}

    # Get the frames per second (fps) of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    start_frame = start_second * fps

    # Initialize variables for saving video every 10 seconds
    interval = 5  # Interval in seconds
    frame_interval = interval * fps  # Number of frames in the interval
    output_counter = 1  # Counter to track the output files
    end_frame = start_frame + frame_interval

    # Define the codec and initialize the VideoWriter as None
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = None
    output_filename = (
        f"video_chunks/output_chunk_st_{start_second}_end_{start_second + interval}.mp4"
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_num > end_frame:
            break

        if frame_num >= start_frame:
            full_image = Image.fromarray(np.uint8(frame))

            # Every step frame refresh values
            if frame_num % step == 0:
                predicted_spots = predict(
                    full_image,
                    bndbox_values=bndbox_values,
                    video=True,
                    batch_size=16,
                    threshhold=0.6,
                )
                spots.update(predicted_spots)

            # this frame is the image, for loop over all the spots
            draw_rectangles_on_batches(
                image=frame, bndbox_values=bndbox_values, spots=spots
            )
            # Initialize VideoWriter when starting a new 10-second interval
            if out is None:
                out = cv2.VideoWriter(
                    output_filename, fourcc, fps, (frame.shape[1], frame.shape[0])
                )

            # Write the frame to the current video file
            out.write(frame)

        frame_num += 1
    if out is not None:
        out.release()

    cap.release()

    return output_filename


def gen_streaming_frames(video_path: str, step: int, bndbox_values: dict):
    cap = cv2.VideoCapture(video_path)
    ret = True
    frame_num = 0
    spots = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        full_image = Image.fromarray(np.uint8(frame))

        # Every step frame refresh values
        if frame_num % step == 0:
            predicted_spots = predict(
                full_image,
                bndbox_values=bndbox_values,
                video=True,
                batch_size=16,
                threshhold=0.6,
            )
            spots.update(predicted_spots)

        # this frame is the image, for loop over all the spots
        draw_rectangles_on_batches(
            image=frame, bndbox_values=bndbox_values, spots=spots
        )
        frame_num += 1

        # Encode the frame to JPEG format
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        # Use a multipart response to stream the frame
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cap.release()


# gen_video_chunks(video_path, step=60, bndbox_values=bndbox_values, start_second=10)
