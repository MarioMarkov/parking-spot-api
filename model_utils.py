from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import torch.quantization

import torch.ao.quantization.qconfig_mapping
from torch.ao.quantization import (
    QConfigMapping,
    get_default_qconfig_mapping,
)
import torch.ao.quantization.quantize_fx as quantize_fx
import copy

# Transformations
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class BatchImages(Dataset):
    def __init__(self, batch_of_spots, full_image, transform):
        self.batch_of_spots = batch_of_spots
        self.patches_batch = []
        self.transforms = transform
        self.full_image = full_image
        for _, spot_info in batch_of_spots.items():
            # Extract coordinates from the bounding box
            xmin = int(spot_info["xmin"])
            ymin = int(spot_info["ymin"])
            xmax = int(spot_info["xmax"])
            ymax = int(spot_info["ymax"])
            # Crop patch for the image
            patch = full_image.crop((xmin, ymin, xmax, ymax))
            self.patches_batch.append(patch)

    def __len__(self):
        return len(self.patches_batch)

    def __getitem__(self, idx):
        image = self.patches_batch[idx]
        tranformed = self.transforms(image)
        return tranformed


class mAlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_channel = 3
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

        self.layer5 = nn.Linear(in_features=48, out_features=1)
        
    def forward(self, x):
        # Convolutions
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Classificator
        x = x.reshape(x.size(0), -1)
        x = self.layer4(x)
        return self.layer5(x)


def fuse_model(model):
    modules_to_fuse = [
        ["layer1.0", "layer1.1"],
        ["layer2.0", "layer2.1"],
        ["layer3.0", "layer3.1"],
        ["layer4.0", "layer4.1"],
    ]

    model_fused = torch.quantization.fuse_modules(model, modules_to_fuse, inplace=False)

    return model_fused


def dynamic_quantize_model(model):
    torch.backends.quantized.engine = "qnnpack"
    model_to_quantize = copy.deepcopy(model)
    model_to_quantize.eval()
    qconfig_mapping = QConfigMapping().set_global(
        torch.ao.quantization.default_dynamic_qconfig
    )

    # prepare
    model_prepared = quantize_fx.prepare_fx(
        model_to_quantize, qconfig_mapping, torch.rand((1, 3, 224, 224)).cpu()
    )
    # no calibration needed when we only have dynamic/weight_only quantization
    # quantize
    model__dynamic_quantized = quantize_fx.convert_fx(model_prepared)

    return model__dynamic_quantized


# def static_quantize_model(model):
#     model_to_quantize = copy.deepcopy(model)
#     #qconfig_mapping = get_default_qconfig_mapping("qnnpack")
#     model_to_quantize.eval()
#     # prepare
#     example_inputs = torch.rand(1, 3, 224, 224)
#     model_prepared = quantize_fx.prepare_fx(
#         model_to_quantize, qconfig_mapping, example_inputs
#     )
#     #torch.backends.quantized.engine = "qnnpack"
#     model_quantized = quantize_fx.convert_fx(model_prepared)
#     return model_quantized


def extract_bndbox_values(tree):
    root = tree.getroot()
    bndbox_values = {
        f"{obj.find('name').text}{i}": {
            "xmin": float(obj.find("bndbox/xmin").text),
            "ymin": float(obj.find("bndbox/ymin").text),
            "xmax": float(obj.find("bndbox/xmax").text),
            "ymax": float(obj.find("bndbox/ymax").text),
        }
        for i, obj in enumerate(root.findall("object"))
    }
    return bndbox_values
