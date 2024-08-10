import os
import cv2
import torch
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from torch.utils.data import DataLoader
import time
import onnxruntime
from utils.model_utils import (
    mAlexNet,
    BatchImages,
    transform,
)


device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")

print("Device is:", device)

model_path = "models/m_alex_net_both_best_acc.pth"

model = mAlexNet().to(device)

model.load_state_dict(
    torch.load(model_path, map_location=torch.device(device), weights_only=True)
)
model.eval()


def predict(
    full_image,
    bndbox_values,
    video: bool = False,
    batch_size: int = 8,
    threshhold: float = 0.5,
):
    start_time = time.time()
    # Convert RGB to BGR
    if not video:
        image_to_draw = cv2.cvtColor(np.array(full_image), cv2.COLOR_RGB2BGR)

    # Every key is one spot
    all_spots_keys = list(bndbox_values.keys())

    num_keys = len(all_spots_keys)
    batch_size = batch_size

    # Make dictionary with every spot as key and value which will be the prediction
    spots_preds = {}
    model_preds = 0

    all_batches = []
    all_batches_with_keys = []

    # Iterate over batches
    start_time = time.time()
    for batch_start_idx in range(0, num_keys, batch_size):
        # end index of the batch
        batch_end_idx = min(batch_start_idx + batch_size, num_keys)

        # Get one bacth of spots in a dictionary form
        batch_with_keys = {
            key: bndbox_values[key]
            for key in all_spots_keys[batch_start_idx:batch_end_idx]
        }

        ds = BatchImages(batch_with_keys, full_image, transform)

        dl = DataLoader(dataset=ds, batch_size=batch_size, shuffle=False, num_workers=0)

        batch = next(iter(dl)).to(device)

        all_batches.append(batch)
        all_batches_with_keys.append(batch_with_keys)

    for b, b_k in zip(all_batches, all_batches_with_keys):
        with torch.no_grad():
            outputs = model(b)
            model_preds += 1
            preds = (torch.sigmoid(outputs) > threshhold).int()

        spots_preds_batch = {
            key: pred.item() for (key, _), pred in zip(b_k.items(), preds)
        }
        spots_preds.update(spots_preds_batch)

    if not video:
        # draw the green and red rectangles
        draw_rectangles_on_batches(
            image=image_to_draw, bndbox_values=bndbox_values, spots=spots_preds
        )

    print("Prediction time: %s seconds" % (time.time() - start_time))
    print("Passes throgh the model: ", model_preds)
    if not video:
        return image_to_draw
    else:
        return spots_preds


def draw_rectangles_on_batches(image, bndbox_values, spots):

    for key, pred in spots.items():
        bndbox = bndbox_values[key]
        xmin = int(bndbox["xmin"])
        ymin = int(bndbox["ymin"])
        xmax = int(bndbox["xmax"])
        ymax = int(bndbox["ymax"])

        color = (0, 0, 255) if pred == 1 else (0, 255, 0)

        cv2.rectangle(
            image,
            (xmin, ymin),
            (xmax, ymax),
            color,
            2,
        )
