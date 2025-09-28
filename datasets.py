import os
import pandas as pd
import numpy as np
from PIL import Image
import pydicom
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def transformSequence(normalize = False, resize = None):
    transforms_list = []

    if resize is not None:
        transforms_list.append(transforms.Resize((resize, resize)))
    if normalize:
        transforms_list.append(transforms.Normalize((0.5,), (0.5,)))

    transforms_list.append(transforms.ToTensor())

    return transforms.Compose(transforms_list)


class ImageLabelDataset(Dataset):
    def __init__(self, csv_file, root, transform=transforms.ToTensor(), to_classify=False):
        """
        csv_file: path to CSV containing [path, label] columns
        img_root: folder where images are stored
        """
        self.data = pd.read_csv(csv_file)
        self.img_root = root
        self.transform = transform
        self.to_classify = to_classify
        self.label_map = {label: idx for idx, label in enumerate(self.data.iloc[:,1].unique())}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        rel_path = self.data.iloc[idx, 0]   # image path (relative or full)
        label = self.data.iloc[idx, 1] # label

        if(self.to_classify):
            label = self.label_map[label]

        filepath = os.path.join(self.img_root, rel_path)

        if filepath.lower().endswith((".png", ".jpg", ".jpeg")):
            img = Image.open(filepath).convert("L")
            # img = (np.array(img)).astype('uint8')
            img_tensor = self.transform(img)

        elif filepath.lower().endswith(".dcm"):
            dicom = pydicom.dcmread(filepath)
            img_array = dicom.pixel_array.astype("uint8")
            img_array = img_array / img_array.max() if img_array.max() > 0 else img_array
            img_tensor = torch.tensor(img_array).unsqueeze(0)  # [1,H,W]

            if self.transform:
                img_tensor = self.transform(transforms.ToPILImage()(img_tensor))

        else:
            raise ValueError(f"Unsupported file format: {filepath}")

        return img_tensor.float(), torch.FloatTensor(label)

class ImageMaskDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, mask_transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.class_map = {
            0: 0,     # outside body
            85: 1,    # heart
            170: 2,   # outside lung field
            255: 3    # lung field
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # Open chest X-ray (BMP → grayscale)
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        # Convert mask values (0,85,170,255) → (0,1,2,3)
        mask = np.array(mask)

        mask_mapped = np.zeros_like(mask, dtype=np.uint8)
        for k, v in self.class_map.items():
            mask_mapped[mask == k] = v

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        if self.mask_transform:
            mask_mapped = self.mask_transform(Image.fromarray(mask_mapped))
        else:
            mask_mapped = torch.from_numpy(mask_mapped, dtype=torch.long)

        return image, mask_mapped