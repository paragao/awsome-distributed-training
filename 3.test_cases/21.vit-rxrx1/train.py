import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
from torchvision import transforms
import pandas as pd

# Define the dataset class
class RxRx1Dataset(Dataset):
    def __init__(self, metadata_df,image_paths, labels, transform=None):
        self.metadata_df = metadata_df
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.metadata_df.shape[0])

    def image_path(experiment,
               plate,
               address,
               site,
               channel,
               base_path='/fsxl/awsankur/rxrx1/rxrx1/images'):
    """
    Returns the path of a channel image.

    Parameters
    ----------
    dataset : str #ignored for now
        what subset of the data: train, test
    experiment : str
        experiment name
    plate : int
        plate number
    address : str
        plate address
    site : int
        site number
    channel : int
        channel number
    base_path : str
        the base path of the raw images

    Returns
    -------
    str the path of image
    """
    return os.path.join(base_path, experiment, "Plate{}".format(plate),
                        "{}_s{}_w{}.png".format(address, site, channel))

    def __getitem__(self, idx):

        experiment = self.metadata_df['experiment'][idx]
        plate = self.metadata_df['plate'][idx]
        well = self.metadata_df['well'][idx]
        site = self.metadata_df['site'][idx]

        channels = (1, 2, 3, 4, 5, 6)

        channel_paths = [image_path(experiment, plate, well, site, c) for c in channels]

        # https://www.rxrx.ai/rxrx1#Download
        six_channel_image_data = np.ndarray(shape=(512, 512, 6), dtype=dtype)

        for ix, img_path in enumerate(channel_paths):
            six_channel_image_data[:, :, ix] = imageio.v3.imread(img_path)

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        # Normalize the image
        image = image / 65535.0  # Normalize 16-bit images
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # Move channels to the first dimension

        return six_channel_image_data, label

# Load data and create dataset

metadata_df = pd.read_csv('/fsxl/awsankur/rxrx1/rxrx1/metadata.csv')

image_paths = [...]  # List of paths to your images
labels = [...]  # List of corresponding labels
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
])

dataset = RxRx1Dataset(metadata_df,image_paths, labels, transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Define the custom ViT model for 6 channels
class CustomViT(nn.Module):
    def __init__(self, base_model, num_classes):
        super(CustomViT, self).__init__()
        self.base_model = base_model
        self.base_model.patch_embed.proj = nn.Conv2d(6, 768, kernel_size=(16, 16), stride=(16, 16))
        self.base_model.head = nn.Linear(768, num_classes)

    def forward(self, x):
        return self.base_model(x)

model = CustomViT(timm.create_model('vit_base_patch16_224', pretrained=True), num_classes=1108)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(10):  # Number of epochs
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Epoch {epoch+1}, Validation Accuracy: {accuracy:.4f}')