# Brain Tumor Segmentation using UNET

This module implements a **UNET-based deep learning model** for **brain tumor segmentation** from MRI images.  
It is part of the **Final Year Project on Medical Image Segmentation**.

---

## 🧠 Model Description

- Architecture: **UNET**
- Task: Brain tumor segmentation
- Input: MRI brain images
- Output: Segmentation masks highlighting tumor regions

---

## 📂 Folder Structure

```text
Brain_UNET/
│
├── code/
│   ├── train.py
│   ├── test.py
│   └── model.py
│
├── results/
│   └── sample_outputs/
│
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
⚙️ Installation
Install required libraries:

bash
Copy code
pip install torch torchvision numpy opencv-python matplotlib scikit-learn
🚀 How to Run
Training
Navigate to the code/ directory:

bash
Copy code
cd Brain_UNET/code
python train.py
Testing / Inference
bash
Copy code
python test.py
📊 Results
Predicted segmentation masks are saved in the results/ folder

Evaluation metrics include:

Dice Coefficient

Intersection over Union (IoU)

Sample outputs are provided for reference.

💾 Trained Model Weights
Trained .pth files are not included due to GitHub file size limits.
They can be shared externally if required.

📄 License
Academic use only.
