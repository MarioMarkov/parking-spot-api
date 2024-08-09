import os
import cv2
import torch
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from torch.utils.data import DataLoader
import time
import onnxruntime
from model_utils import (
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
    torch.load(model_path, map_location=torch.device(device))
)
model.eval()


def predictv2(full_image, bndbox_values):
    start_time = time.time()
    # Convert RGB to BGR
    image_to_draw = cv2.cvtColor(np.array(full_image), cv2.COLOR_RGB2BGR)
    print("Data Tranformation time: %s seconds" % (time.time() - start_time))

    # Every key is one spot
    all_spots_keys = list(bndbox_values.keys())

    num_keys = len(all_spots_keys)
    batch_size = 8

    # Make dictionary with every spot as key and value which will be the prediction
    spots_preds = {}
    model_preds = 0

    # Iterate over batches
    start_time = time.time()
    for batch_start_idx in range(0, num_keys, batch_size):
        # end index of the batch
        batch_end_idx = min(batch_start_idx + batch_size, num_keys)

        # Get one bacth of spots in a dictionary form
        batch_of_spots = {key: bndbox_values[key] for key in all_spots_keys[batch_start_idx:batch_end_idx]}

        ds = BatchImages(batch_of_spots, full_image, transform)

        dl = DataLoader(dataset=ds, batch_size=batch_size, shuffle=False, num_workers=0)

        batch = next(iter(dl)).to(device)

        with torch.no_grad():
            outputs = model(batch)
            model_preds += 1
            preds = (torch.sigmoid(outputs) > 0.5).int()

        # create batch of {'spot0': 0, ...}
        spots_preds_batch = {
            key: pred.item() for (key, _), pred in zip(batch_of_spots.items(), preds)
        }
        spots_preds.update(spots_preds_batch)

    print("spots_preds: ", spots_preds)
    # draw the green and red rectangles 
    draw_rectangles_on_batches(spots_preds=spots_preds, bndbox_values=bndbox_values, image_to_draw=image_to_draw)
    
    print("Prediction time: %s seconds" % (time.time() - start_time))
    print("Model predictions: ", model_preds)
    return image_to_draw


def draw_rectangles_on_batches(spots_preds, bndbox_values, image_to_draw):

    for key, pred in spots_preds.items():
        bndbox = bndbox_values[key]
        xmin = int(bndbox["xmin"])
        ymin = int(bndbox["ymin"])
        xmax = int(bndbox["xmax"])
        ymax = int(bndbox["ymax"])

        color = (0, 0, 255) if pred == 1 else (0, 255, 0)

        cv2.rectangle(
            image_to_draw,
            (xmin, ymin),
            (xmax, ymax),
            color,
            2,
        )
