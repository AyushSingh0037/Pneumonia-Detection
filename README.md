🫁 Pneumonia Detection from Chest X-ray

Deep learning system for classifying Chest X-ray images into NORMAL vs PNEUMONIA using a transfer-learning based ResNet-18 architecture implemented with PyTorch, and an inference UI built with Streamlit.

📂 Dataset

This project uses the Chest X-Ray Pneumonia Dataset from Kaggle:
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

Dataset Structure:

chest_xray/
├── train/
├── val/
└── test/


Classes:

NORMAL (label = 0)

PNEUMONIA (label = 1)

🧠 Model Architecture

Base model: ResNet-18 pretrained on ImageNet.

Modifications:

model.fc = nn.Sequential(
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 2)
)


Training strategy:

Transfer learning (frozen backbone)

Early stopping on validation loss

Cross-entropy loss

Adam optimizer

📊 Evaluation Metrics

The model reports:

✔ Accuracy
✔ Precision
✔ Recall
✔ F1-score
✔ Confusion Matrix
✔ Classification Report

Example output:

Accuracy: 0.93
Precision: 0.94
Recall: 0.92
F1-score: 0.93

🖥 Training

To train:

python -m src.Train


Weights are saved to:

weights/best_model.pth

🔍 Evaluation

To evaluate the best model:

python -m src.evaluate

🧪 Inference (Programmatic)
from src.inference import load_model, predict_image

model = load_model("weights/best_model.pth")
result = predict_image("sample_xray.jpg", model)
print(result)


Output:

{'prediction': 'PNEUMONIA', 'confidence': 97.3}

🖥 Streamlit Application

Run locally:

streamlit run app.py


This opens an interface where you can upload an X-ray and get predictions.

☁️ Deployment (Optional)

You can deploy on:

Streamlit Cloud

Push repo to GitHub

Go to https://share.streamlit.io

Connect repo

Select app.py

Deploy

HuggingFace Spaces

Supports:

streamlit

gradio

📦 Requirements

Create virtual env and install deps:

pip install -r requirements.txt


Minimal requirements:

torch
torchvision
numpy
scikit-learn
pillow
streamlit

📁 Project Structure
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
├── README.md
└── requirements.txt

📌 Notes

⚠️ Dataset not included — download from Kaggle and place under data/
⚠️ Weights not included — generated after training

📜 License

MIT License – Free for academic and commercial use.
