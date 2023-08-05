import os
import cv2
import torch
import numpy

from PIL import Image
from torchvision import transforms
import xml.etree.ElementTree as ET


device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")


def extract_bndbox_values(tree):
    root = tree.getroot()
    bndbox_values = {}

    for i, obj in enumerate(root.findall("object")):
        bndbox = obj.find("bndbox")
        name = obj.find("name").text

        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        bndbox_values[name + str(i)] = {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        }

    return bndbox_values


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
        open_cv_image = numpy.array(pil_image)
        # Convert RGB to BGR
        image_to_draw = open_cv_image[:, :, ::-1].copy()

    print("Full image:", full_image)
    print("OpenCV image:", full_image)

    # Transformations
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    model = torch.load("full_256_alex_net_pk_lot.pth", map_location=device)
    model.eval()
    model.to(device)

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
    if require_parsing:
        cv2.imwrite("predicted_image.jpg", image_to_draw)
        return "Success"

    return image_to_draw
