# 🫁 Pneumonia Detection from Chest X-ray

Deep learning system for classifying Chest X-ray images into **NORMAL** vs **PNEUMONIA** using a transfer-learning based **ResNet-18** architecture implemented with **PyTorch**, and an inference UI built with **Streamlit**.

---

## 📂 Dataset

This project uses the **Chest X-Ray Pneumonia Dataset** from Kaggle:  
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

**Dataset structure:**
chest_xray/
├── train/
├── val/
└── test/

**Classes:**
- NORMAL (label = 0)
- PNEUMONIA (label = 1)

---

## 🧠 Model Architecture

Base model: **ResNet-18** pretrained on ImageNet.

**Modifications:**
```python
model.fc = nn.Sequential(
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 2)
)
```

---
## 📊 Evaluation Results
The model achieved the following performance on the validation set:
Metric	Score
Accuracy	93.75%
Precision	88.89%
Recall	100%
F1-Score	94.12%

---
## Training strategy:

Transfer learning (frozen backbone)
Early stopping on validation loss
Cross-entropy loss
Adam optimizer

---
## Confusion Matrix

            Pred Normal | Pred Pneumonia
True Normal      7      |       1
True Pneumonia   0      |       8


The model shows high sensitivity to pneumonia cases (Recall = 100%), which is desirable in medical screening applications where false negatives are costly.

## 🖥 Training
python -m src.Train


## Weights saved to:
weights/best_model.pth

## 🔍 Evaluation
python -m src.evaluate

# Pneumonia Detection from Chest X-ray

A deep learning system for detecting pneumonia from chest X-ray images using transfer learning with a ResNet-18 backbone implemented in PyTorch. Includes a Streamlit-based inference UI for real-time prediction.

---

## Evaluation Results

Validation set results:

| Metric     | Score    |
|------------|----------|
| Accuracy   | 93.75%   |
| Precision  | 88.89%   |
| Recall     | 100%     |
| F1-Score   | 94.12%   |

Confusion Matrix:

```
                 Pred NORMAL | Pred PNEUMONIA
True NORMAL          7       |        1
True PNEUMONIA       0       |        8
```

Clinical note:  
The model achieves high recall for pneumonia (100%), which aligns with medical screening requirements where false negatives are costly.

---

## Training

Run training:

```bash
python -m src.Train
```

Best weights saved to:

```
weights/best_model.pth
```

---

## Evaluation

Run evaluation:

```bash
python -m src.evaluate
```

---

## Inference (Programmatic)

```python
from src.inference import load_model, predict_image

model = load_model("weights/best_model.pth")
result = predict_image("sample_xray.jpg", model)
print(result)
```

Example output:

```
{'prediction': 'PNEUMONIA', 'confidence': 97.3}
```

---

## Streamlit Application

Run interactive UI:

```bash
streamlit run app.py
```

Deployment options:
- Streamlit Cloud
- HuggingFace Spaces

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Core packages:
- torch
- torchvision
- numpy
- scikit-learn
- pillow
- streamlit

---

## Project Structure

```
Pneumonia-Detection/
├── src/
│   ├── Train.py
│   ├── Validation.py
│   ├── model.py
│   ├── inference.py
│   ├── evaluate.py
│   └── dataset.py
├── weights/
│   └── best_model.pth
├── app.py
├── requirements.txt
└── README.md
```

---

## Notes

- Dataset must be downloaded manually from Kaggle.
- Weights are generated after running training.
- Streamlit UI is optional but useful for demonstration.

---

## License

MIT License

---
## 🌍 Live Demo

Try the live model here:  
https://huggingface.co/spaces/AyushSingh0037/pneumonia-xray-detector

