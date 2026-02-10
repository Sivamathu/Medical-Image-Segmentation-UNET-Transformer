
# Liver Tumor Segmentation using UNET

This module implements a **UNET-based model** for **liver tumor segmentation** from CT images.  
It is developed as part of a **Final Year Project**.

---

## 🧠 Model Description

- Architecture: **UNET**
- Task: Liver tumor segmentation
- Input: CT scan images
- Output: Tumor segmentation masks

---

## 📂 Folder Structure

```text
Liver_UNET/
│
├── source_code/
│   ├── train.py
│   ├── test.py
│   └── model.py
│
├── results/
│   └── sample_outputs/
│
└── README.md
📂 Dataset
Dataset files are not included in this repository.

🔗 Dataset Link
Liver Tumor Dataset (LiTS – Kaggle):
https://www.kaggle.com/datasets/ag3ntsp1d3rx/litsdataset2

📁 Dataset Directory Format
text
Copy code
dataset/
├── images/
└── masks/
⚙️ Installation
Install required dependencies:

bash
Copy code
pip install torch torchvision numpy opencv-python matplotlib scikit-learn
🚀 How to Run
Training
bash
Copy code
cd Liver_UNET/source_code
python train.py
Testing / Inference
bash
Copy code
python test.py
📊 Results
Segmentation outputs are saved in the results/ directory

Performance evaluated using Dice Score and IoU

💾 Trained Model Weights
Trained model weights are not pushed to GitHub due to size constraints.
They are available externally if required.

📄 License
Academic use only.
