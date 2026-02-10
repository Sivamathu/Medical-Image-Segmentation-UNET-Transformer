# Brain Tumor Segmentation using UNET + Transformer

This module implements a **hybrid UNET + Transformer architecture** for improved **brain tumor segmentation** performance.

---

## 🧠 Model Description

- Architecture: **UNET + Transformer Encoder**
- Task: Brain tumor segmentation
- Purpose: Improve global feature representation compared to standard UNET

---

## 📂 Folder Structure

```text
Brain_UNET_Transformer/
│
├── code/
│   ├── train.py
│   ├── test.py
│   ├── unet.py
│   └── transformer.py
│
├── requirements.txt
└── README.md
📂 Dataset
The dataset is not included in this repository.

🔗 Dataset Link
Brain Tumor Dataset (BraTS – Kaggle):
https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation

📁 Dataset Directory Format
text
Copy code
dataset/
├── images/
└── masks/
⚙️ Installation & Requirements
Create and activate a virtual environment (optional):

bash
Copy code
python -m venv venv
venv\Scripts\activate   # Windows
Install dependencies:

bash
Copy code
pip install -r requirements.txt
🚀 How to Run
Training
bash
Copy code
cd Brain_UNET_Transformer/code
python train.py
Testing / Inference
bash
Copy code
python test.py
📊 Results
Output masks and evaluation metrics are generated after training

Improved segmentation accuracy compared to baseline UNET

💾 Trained Model Weights
Model weights (.pth) are excluded due to GitHub size limitations.
They can be shared via external storage.

📄 License
Academic and research use only.
