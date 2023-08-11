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
    dynamic_quantize_model,
    fuse_model,
    mAlexNet,
    BatchImages,
    static_quantize_model,
    transform,
)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
    if torch.backends.mps.is_available()
    else "cpu"
)
print("Device is:", device)
model_path = "m_alex_net.pth"


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


print("Loading model...")

# ort_session = onnxruntime.InferenceSession("onxx_malex_net.onnx")

model = mAlexNet(num_classes=2).to(device)
# model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
model = static_quantize_model(model)
model.load_state_dict(
    torch.load("model_qunatized_static.pt", map_location=torch.device(device))
)

# model = fuse_model(model)
# model = dynamic_quantize_model(model)
# model = torch.jit.script(model)
model.eval()
print(model)


def predictv2(full_image, bndbox_values):
    start_time = time.time()
    # pil_image = full_image.convert("RGB")
    image_to_draw = cv2.cvtColor(np.array(full_image), cv2.COLOR_RGB2BGR)
    # Convert RGB to BGR
    # image_to_draw = np.array(pil_image)
    # image_to_draw[:, :, [0, 2]] = image_to_draw[:, :, [2, 0]]
    print("Data Tranformation time: %s seconds" % (time.time() - start_time))

    # Every key is one spot
    all_spots_keys = list(bndbox_values.keys())
    num_keys = len(all_spots_keys)
    batch_size = 16

    # Make dictionary with every spot as key and value which will be the prediction
    spots_preds = {}
    model_preds = 0

    # Iterate over batches
    start_time = time.time()
    for start in range(0, num_keys, batch_size):
        # end index of the batch
        end = min(start + batch_size, num_keys)

        # Get one bacth of spots in a dictionary form
        batch_of_spots = {key: bndbox_values[key] for key in all_spots_keys[start:end]}

        ds = BatchImages(batch_of_spots, full_image, transform)

        dl = DataLoader(dataset=ds, batch_size=batch_size, shuffle=False, num_workers=0)

        batch = next(iter(dl))

        with torch.no_grad():
            outputs = model(batch)
            model_preds += 1
            _, preds = torch.max(outputs, dim=1)

        spots_preds_batch = {
            key: pred.item() for (key, _), pred in zip(batch_of_spots.items(), preds)
        }
        spots_preds.update(spots_preds_batch)

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
    print("Prediction time: %s seconds" % (time.time() - start_time))
    print("Model predictions: ", model_preds)
    return image_to_draw


def predict(image_path, xml_dir, require_parsing):
    if require_parsing:
        tree = ET.parse(os.path.join("annotations", xml_dir))
        bndbox_values = extract_bndbox_values(tree)
        full_image = Image.open(image_path)
        image_to_draw = cv2.imread(image_path)
    else:
        full_image = image_path
        bndbox_values = extract_bndbox_values(xml_dir)
        pil_image = full_image.convert("RGB")
        open_cv_image = np.array(pil_image)
        # Convert RGB to BGR
        image_to_draw = open_cv_image[:, :, ::-1].copy()

    start_time = time.time()
    for key in bndbox_values:
        values = bndbox_values[key]
        # Extract coordinates from the bounding box
        xmin = int(values["xmin"])
        ymin = int(values["ymin"])
        xmax = int(values["xmax"])
        ymax = int(values["ymax"])
        # Crop patch for the image
        patch = full_image.crop((xmin, ymin, xmax, ymax))

        img = transform(patch)
        img = img.unsqueeze(0)
        img = img.to(device)
        with torch.no_grad():
            outputs = model(img)
            _, preds = torch.max(outputs, 1)
            is_busy = preds[0]

        # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
        # ort_outs = ort_session.run(None, ort_inputs)
        # img_out_y = ort_outs[0]
        # preds = list()
        # [preds.append(np.argmax(pred)) for pred in img_out_y]

        if is_busy == 1:
            # Busy 1 Red
            cv2.rectangle(
                image_to_draw,
                (int(xmin), int(ymin)),
                (int(xmax), int(ymax)),
                (0, 0, 255),
                2,
            )

        else:
            # Free 0 green
            cv2.rectangle(
                image_to_draw,
                (int(xmin), int(ymin)),
                (int(xmax), int(ymax)),
                (0, 255, 0),
                2,
            )
    print("Prediction time: %s seconds" % (time.time() - start_time))
    if require_parsing:
        cv2.imwrite("predicted_image.jpg", image_to_draw)
        return "Success"

    return image_to_draw
