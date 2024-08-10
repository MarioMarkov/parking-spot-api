import cv2
import torch
import numpy as np
from PIL import Image

from utils.inference_utils import draw_rectangles_on_batches, predict
from utils.model_utils import (
    extract_bndbox_values,
    mAlexNet,
)


device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")

print(f"Using {device}")

video_path = "./example/parking_video.mp4"
annotation_path = "./example/pg_survailance.xml"
model_path = "./models/m_alex_net_both_best_acc.pth"


model = mAlexNet().to(device)

model.load_state_dict(
    torch.load(model_path, map_location=torch.device(device), weights_only=True)
)
model.eval()


## Inference video

cap = cv2.VideoCapture(video_path)
ret = True
frame_num = 0
step = 60
bndbox_values = extract_bndbox_values(annotation_path)
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
    draw_rectangles_on_batches(image=frame, bndbox_values=bndbox_values, spots=spots)
    frame_num += 1

    cv2.imshow("Parking Lot", frame)
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
