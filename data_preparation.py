# data_preparation.py
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Optional
import numpy as np

def pil_loader(fp: Path, mode: str) -> Image.Image:
    with open(fp, "rb") as f:
        img = Image.open(f)
        return img.convert(mode)

def first_and_last_nonzeros(arr):
    for i in range(len(arr)):
        if arr[i] != 0:
            break
    left = i
    for i in reversed(range(len(arr))):
        if arr[i] != 0:
            break
    right = i
    return left, right

def crop(filename: Path, padding: int = 8) -> Optional[Image.Image]:
    image = pil_loader(filename, mode="RGBA")

    # Replace the transparency layer with a white background
    new_image = Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, (0, 0), image)
    new_image = new_image.convert("L")

    # Invert the color to have a black background and white text
    arr = 255 - np.array(new_image)

    # Area that has text should have nonzero pixel values
    row_sums = np.sum(arr, axis=1)
    col_sums = np.sum(arr, axis=0)
    y_start, y_end = first_and_last_nonzeros(row_sums)
    x_start, x_end = first_and_last_nonzeros(col_sums)

    # Some images have no text
    if y_start >= y_end or x_start >= x_end:
        print(f"{filename.name} is ignored because it does not contain any text")
        return None

    # Cropping
    cropped = arr[y_start : y_end + 1, x_start : x_end + 1]
    H, W = cropped.shape

    # Add paddings
    new_arr = np.zeros((H + padding * 2, W + padding * 2))
    new_arr[padding : H + padding, padding : W + padding] = cropped

    # Invert the color back to have a white background and black text
    new_arr = 255 - new_arr
    return Image.fromarray(new_arr).convert("L")

home = Path.home()
RAW_IMAGES_DIRNAME = home / "Downloads/ML_Project/formula_images"
PROCESSED_IMAGES_DIRNAME = home / "Downloads/ML_Project/formula_images_processed"

class LatexDataset(Dataset):
    def __init__(self, data_dir, transform=None, split="train"):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        self.images = []
        self.labels = []

        self.load_data()

    def load_data(self):
        formula_file = os.path.join(self.data_dir, "im2latex_formulas.lst")
        image_list_file = os.path.join(self.data_dir, f"im2latex_{self.split}.lst")

        with open(formula_file, "r", encoding="latin-1") as f:
            formulas = f.readlines()

        with open(image_list_file, "r", encoding="latin-1") as f:
            image_list = f.readlines()

        for line in image_list:
            parts = line.strip().split(" ")
            formula_idx = int(parts[0])
            image_name = parts[1]
            render_type = parts[2]

            label = formulas[formula_idx].strip()

            image_path = os.path.join(self.data_dir, "formula_images", f"{image_name}.png")

            self.images.append(image_path)
            self.labels.append(label)
        
        if not PROCESSED_IMAGES_DIRNAME.exists():
            PROCESSED_IMAGES_DIRNAME.mkdir(parents=True, exist_ok=True)
            print("Cropping images...")
            for image_filename in RAW_IMAGES_DIRNAME.glob("*.png"):
                cropped_image = crop(image_filename, padding=8)
                if not cropped_image:
                    continue
                cropped_image.save(PROCESSED_IMAGES_DIRNAME / image_filename.name)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return {"image": image, "label": label}

transform = transforms.Compose([
    transforms.ToTensor(),
])

data_dir = r"C:\Users\avnee\Downloads\ML_Project\formula_images"

train_dataset = LatexDataset(data_dir, transform=transform, split="train")
val_dataset = LatexDataset(data_dir, transform=transform, split="validate")
test_dataset = LatexDataset(data_dir, transform=transform, split="test")

batch_size = 32
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
