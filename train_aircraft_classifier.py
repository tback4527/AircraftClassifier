import os
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import AircraftClassifier

now = datetime.datetime.now

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device : {device}")


##############################################################
# Image Processing
##############################################################
def image_loader(folder_path):
    images = []
    labels = []
    scene_ids = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert("RGB")
            img = ImageEnhance.Contrast(img).enhance(1.5)
            img_array = np.array(img)
            images.append(img_array)
            labels.append(get_image_labels(filename))
            scene_ids.append(get_scene_ids(filename))
    return np.array(images), np.array(labels), np.array(scene_ids)


def get_image_labels(filename):
    return filename.split("__")[0]


def get_scene_ids(filename):
    return filename.split("__")[1]


def display_image_label(images, labels, preds, scenes):
    idx = random.randint(0, len(images) - 1)
    img = images[idx]

    if labels[idx] == 0:
        label = "No or partial aircraft detected."
    elif labels[idx] == 1:
        label = "Aircraft present in image."
    else:
        label = None

    if labels[idx] == preds[idx]:
        class_result = "Classified correctly."
    else:
        class_result = "Misclassified."

    scene = scenes[idx]

    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Scene ID: {scene} : label = {class_result}")

    plt.show()


##############################################################
# Dataset
##############################################################


class AircraftDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


##############################################################
# Model
##############################################################


# class AircraftClassifier(nn.Module):
#     def __init__(self, in_channels, filters=32, kernel_size=3, pool_size=2):
#         super().__init__()

#         self.conv1 = nn.Conv2d(in_channels, filters, kernel_size)
#         self.pool1 = nn.MaxPool2d(pool_size)
#         self.conv2 = nn.Conv2d(filters, filters, kernel_size)
#         self.pool2 = nn.MaxPool2d(pool_size)
#         self.dropout = nn.Dropout(0.25)
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.LazyLinear(128)
#         self.drouput2 = nn.Dropout(0.5)
#         self.fc2 = nn.LazyLinear(1)

#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = self.pool1(x)
#         x = torch.relu(self.conv2(x))
#         x = self.pool2(x)
#         x = self.dropout(x)
#         x = self.flatten(x)
#         x = torch.relu(self.fc1(x))
#         x = self.drouput2(x)
#         x = self.fc2(x)
#         return x


##############################################################
# Loading Images
##############################################################

images_path = "./data/planesnet"
images, labels, scenes = image_loader(images_path)

print(f"Loaded {len(images)} images with shape {images[0].shape}")


##############################################################
# Dataset Prep
##############################################################

images = images.astype("float32")
images /= 255
labels = labels.astype("int64")

(
    train_val_images,
    test_images,
    train_val_labels,
    test_labels,
    train_val_scenes,
    test_scenes,
) = train_test_split(images, labels, scenes, test_size=0.2, random_state=33)

train_images, val_images, train_labels, val_labels, train_scenes, val_scenes = (
    train_test_split(
        train_val_images, train_val_labels, train_val_scenes, test_size=0.25
    )
)

print(f"Training Set: {len(train_images)} images.")
print(f"Validation Set: {len(val_images)} images.")
print(f"Testing Set: {len(test_images)} images.")


##############################################################
# DataLoaders
##############################################################

batch_size = 32
epochs = 40

train_dataset = AircraftDataset(train_images, train_labels)
val_dataset = AircraftDataset(val_images, val_labels)
test_dataset = AircraftDataset(test_images, test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

in_channels = images[0].shape[-1]

filters = 32
pool_size = 2
kernel_size = 3

model = AircraftClassifier(
    in_channels=in_channels,
    filters=filters,
    kernel_size=kernel_size,
    pool_size=pool_size,
).to(device)

with torch.no_grad():
    dummy = next(iter(train_loader))[0].to(device)
    model(dummy)


##############################################################
# TensorBoard setup
##############################################################

log_dir = "logs/fit" + now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir=log_dir)


##############################################################
# Loss / Optimizer
##############################################################

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())


def run_epoch(loader, train=True):
    model.train() if train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            lbls = lbls.unsqueeze(1)

            if train:
                optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, lbls)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == lbls).sum().item()
            total += imgs.size(0)

    return total_loss / total, correct / total


##############################################################
# Model Fitting and Training
##############################################################

t = now()

train_acc_history, val_acc_history = [], []
train_loss_history, val_loss_history = [], []

for epoch in range(1, epochs + 1):
    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss, val_acc = run_epoch(val_loader, train=False)

    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)

    writer.add_scalar("Accuracy/train", train_acc, epoch)
    writer.add_scalar("Accuracy/val", val_acc, epoch)
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)

    print(
        f"Epoch {epoch}/{epochs} - "
        f"    loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - "
        f"    val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}"
    )

writer.close()

print("Training time: %s" % (now() - t))

print(f"Validation Accuracy: {val_acc_history[-1]}")

results_plots = Path("./static/images")
results_plots.mkdir(parents=True, exist_ok=True)

plt.plot(list(range(1, epochs + 1)), train_acc_history)
plt.plot(list(range(1, epochs + 1)), val_acc_history)
plt.legend(["Accuracy", "Validation Accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Accurracy")
plt.title("Image Classification Training Accuracy")
# plt.savefig(f"{results_plots}/acc.png")
plt.savefig(results_plots / "acc.png")
plt.clf()

plt.plot(list(range(1, epochs + 1)), train_loss_history)
plt.plot(list(range(1, epochs + 1)), val_loss_history)
plt.legend(["Loss", "Validation Loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Image Classification Training Loss")
# plt.savefig(f"{results_plots}/loss.png")
plt.savefig(results_plots / "loss.png")
plt.clf()

torch.save(model.state_dict(), "aircraft_class.pt")

# To reload:
#   model = AircraftClassifier(in_channels=in_channels, filters=filters, kernel_size=kernel_size, pool_size=pool_size)
#   model(dummy_batch) # forward once to initialize LazyLinear layers first
#   model.load_state_dict(torch.load('aircraft_class.pt'))


##############################################################
# Testing and Soring on unseen data
##############################################################

test_loss, test_acc = run_epoch(test_loader, train=False)
print(f"Testing Accuracy: {test_acc}")

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy().reshape(-1)
        all_preds.extend(preds.tolist())
        all_labels.extend(lbls.numpy().astype(int).tolist())

pred_labels = np.array(all_preds)
test_labels = np.array(all_labels)

cm = confusion_matrix(test_labels, pred_labels)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["No Aircraft Present", "Aircraft Detected"]
)

disp.plot()
plt.yticks(rotation=45)
plt.title(f"Image Classification Accuracy - {str(round(test_acc * 100, 2))}%")
# plt.savefig(f"{results_plots}/cm.png")
plt.savefig(results_plots / "cm.png")

display_image_label(test_images, test_labels, pred_labels, test_scenes)
