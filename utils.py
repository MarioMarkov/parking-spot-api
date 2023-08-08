import os
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import xml.etree.ElementTree as ET
import time
import onnxruntime
import torch
import torch.nn as nn
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
    if torch.backends.mps.is_available()
    else "cpu"
)
model_path = "m_alex_net.pth"


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


print("loading model...")


class mAlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.input_channel = 3
        self.num_output = num_classes
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_channel,
                out_channels=16,
                kernel_size=11,
                stride=4,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=20, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=30, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.layer4 = nn.Sequential(
            nn.Linear(30 * 3 * 3, out_features=48), nn.ReLU(inplace=True)
        )

        self.layer5 = nn.Sequential(
            nn.Linear(in_features=48, out_features=self.num_output)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.layer4(x)
        logits = self.layer5(x)
        return logits


#ort_session = onnxruntime.InferenceSession("onxx_malex_net.onnx")

model = mAlexNet(num_classes=2).to(device)
model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

model.eval()


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
        open_cv_image = np.array(pil_image)
        # Convert RGB to BGR
        image_to_draw = open_cv_image[:, :, ::-1].copy()

    # Transformations
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
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
            print(outputs)
            _, preds = torch.max(outputs, 1)
        # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
        # ort_outs = ort_session.run(None, ort_inputs)
        # img_out_y = ort_outs[0]
        # preds = list()
        # [preds.append(np.argmax(pred)) for pred in img_out_y]
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
    print("Prediction time: %s seconds" % (time.time() - start_time))
    if require_parsing:
        cv2.imwrite("predicted_image.jpg", image_to_draw)
        return "Success"

    return image_to_draw
