import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from model import src.build_model


def evaluate_best_model(dataloader, device, model_path="best_model.pth"):
    
  model = build_model().to(device)
  model.load_state_dict(torch.load(model_path, map_location=device))
  model.eval()

  all_preds = []
  all_labels = []

  with torch.no_grad():
    for imgs, labels in dataloader:
      imgs = imgs.to(device)
      labels = labels.to(device)

      outputs = model(imgs)
      preds = torch.argmax(outputs, dim=1)

      all_preds.extend(preds.cpu().numpy())
      all_labels.extend(labels.cpu().numpy())

  all_preds = np.array(all_preds)
  all_labels = np.array(all_labels)

  acc = accuracy_score(all_labels, all_preds)
  precision = precision_score(all_labels, all_preds)
  recall = recall_score(all_labels, all_preds)
  f1 = f1_score(all_labels, all_preds)
  cm = confusion_matrix(all_labels, all_preds)

  print("\nClassification Report (Best Model)")
  print(classification_report(
      all_labels,
      all_preds,
      target_names=["NORMAL", "PNEUMONIA"]
  ))

  print("Confusion Matrix")
  print(cm)

  return {
      "accuracy": acc,
      "precision": precision,
      "recall": recall,
      "f1_score": f1,
      "confusion_matrix": cm
  }
