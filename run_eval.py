
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import ChestXrayDataset
from model import build_model

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------
# Transforms
# -------------------------
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------
# Dataset & Loader (THIS IS REQUIRED)
# -------------------------
val_dataset = ChestXrayDataset(
    root_dir="data/chest_xray/val",
    transform=val_transform
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print(f"Val samples: {len(val_dataset)}")

# -------------------------
# Load model (NO training)
# -------------------------
model = build_model().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

print("✅ Best model loaded (NO retraining)")
