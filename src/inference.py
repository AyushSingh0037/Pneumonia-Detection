import torch
from torchvision import transforms
from PIL import Image

from src.model import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inference_transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(
      mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225]
  )
])

def load_model(model_path="weights/best_model.pth"):
  model = build_model().to(device)
  model.load_state_dict(torch.load(model_path, map_location=device))
  model.eval()
  return model

def predict_image(image_path, model):
    
  image = Image.open(image_path).convert("RGB")
  image = inference_transform(image)
  image = image.unsqueeze(0).to(device)  # (1, C, H, W)

  with torch.no_grad():
    outputs = model(image)
    probs = torch.softmax(outputs, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_class].item()

  class_names = ["NORMAL", "PNEUMONIA"]

  return {
    "prediction": class_names[pred_class],
    "confidence": round(confidence * 100, 2)
  }


if __name__ == "__main__":
  model = load_model("best_model.pth")

  image_path = "sample_xray.jpg"  # 👈 change this
  result = predict_image(image_path, model)

  print("Prediction:", result["prediction"])
  print("Confidence:", result["confidence"], "%")
