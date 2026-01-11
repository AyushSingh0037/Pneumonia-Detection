import os
from PIL import Image
from torch.utils.data import Dataset

class ChestXrayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []

        class_map = {"NORMAL": 0, "PNEUMONIA": 1}

        for cls in class_map:
            cls_path = os.path.join(root_dir, cls)
            for img in os.listdir(cls_path):
                if img.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.images.append(os.path.join(cls_path, img))
                    self.labels.append(class_map[cls])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]
