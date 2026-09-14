from flask import Flask, render_template, request

import torch
import numpy as np
from PIL import Image
from pathlib import Path

from model import AircraftClassifier

app = Flask(__name__)

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device : {device}")

in_channels = 3
filters = 32
kernel_size = 3
pool_size = 2

model = AircraftClassifier(
    in_channels=in_channels,
    filters=filters,
    kernel_size=kernel_size,
    pool_size=pool_size,
)

img_rows, img_cols = 20, 20

with torch.no_grad():
    dummy = torch.zeros(1, in_channels, img_rows, img_cols)
    model(dummy)

model.load_state_dict(torch.load("aircraft_class.pt", map_location=device))
model.to(device)
model.eval()


def convert_label(label):
    if label == 0:
        text_label = "No or Partial Aircraft Present"
    elif label == 1:
        text_label = "Full Aircraft Detected"
    else:
        text_label = None
    return text_label


@app.route("/", methods=["GET"])
def index():
    image_path = "aircraft.jpg"
    return render_template("index.html", image_path=image_path)


@app.route("/", methods=["POST"])
def predict():
    image_file = request.files["imagefile"]
    image_path = Path("./static/images")
    image_path.mkdir(parents=True, exist_ok=True)
    image_path = image_path / image_file.filename
    image_file.save(image_path)

    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image).astype("float32") / 255

    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)

    label = int(image_file.filename.split("__")[0])

    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.sigmoid(output)
        pred = (pred > 0.5).int().item()

    image_label = convert_label(label)
    image_pred = convert_label(pred)

    return render_template(
        "index.html",
        label=image_label,
        image_pred=image_pred,
        image_path=image_file.filename,
    )


if __name__ == "__main__":
    app.run(port=3000, debug=True)
