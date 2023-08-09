from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn

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