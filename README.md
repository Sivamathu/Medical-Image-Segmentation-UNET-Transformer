# Medical Image Segmentation using UNET and UNET-Transformer

This repository contains my **Final Year Project** on **Medical Image Segmentation** using deep learning models:
* **UNET**
* **UNET + Transformer** (Hybrid Architecture)

The project focuses on **brain tumor** and **liver tumor** segmentation from medical images to assist in automated diagnosis.

***

## 📌 Project Structure

```text
Medical-Image-Segmentation-UNET-Transformer/
│
├── Brain_UNET/
│   ├── code/
│   └── results/
│
├── Liver_UNET/
│   ├── source_code/
│   └── results/
│
├── Brain_UNET_Transformer/
│   ├── code/
│   └── requirements.txt
│
├── .gitignore
└── README.md

```
## 🧠 Models Implemented

UNET: Applied for brain tumor segmentation.

UNET: Applied for liver tumor segmentation.

UNET + Transformer: A fusion model implemented for improved brain tumor segmentation accuracy.

## 📂 Dataset

Due to large file sizes and privacy restrictions, datasets are not included in this repository.

### Download Links

Brain Tumor Dataset (BraTS)

 [Brain Tumor Dataset](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation?resource=download)

Liver Tumor Dataset

[Liver Tumor Dataset](https://www.kaggle.com/datasets/ag3ntsp1d3rx/litsdataset2)

### Directory Setup

After downloading, please organize the dataset in the following format:

Plaintext
```
dataset/
├── images/
└── masks/
```
## ⚙️ Installation & Requirements

### 1. Create a Virtual Environment (Optional but Recommended)

Bash
```
# Create virtual environment
python -m venv venv
```

```
# Activate (Linux/Mac)
source venv/bin/activate

```

```
# Activate (Windows)
venv\Scripts\activate

```

### 2. Install Dependencies

For UNET + Transformer model:

Bash

```
pip install -r Brain_UNET_Transformer/requirements.txt
If requirements.txt is not available, install manually:

```

Bash

```
pip install torch torchvision numpy opencv-python matplotlib scikit-learn

```
## 🚀 How to Run the Project


### Training

Navigate to the respective code/ folder and run:

Bash

```
# Example training command
python train.py

```

Or if the script is named differently

```
python main.py
```
### Testing / Inference


To run predictions on the test set:

Bash

```
python test.py

```

Or

```
python predict.py

```
## 📊 Results

Segmentation Masks: Saved in the results/ folder.

Metrics: Evaluation metrics such as Dice Score and IoU are computed after training.

Samples: Sample output images are included in the results folder for reference.

#### Input

![Image](https://github.com/user-attachments/assets/55cc9dde-cf52-47e9-8d48-431c57d2497d)

#### Output

<img width="256" height="256" alt="Image" src="https://github.com/user-attachments/assets/3d817bd0-2097-4228-aaf3-313232245f6e" />

## 💾 Trained Model Weights

Trained model files (.pth) are not included due to GitHub file size limits. They can be:

Shared via Google Drive.

Provided upon request.

## 🧪 Technologies Used

Language: Python

Framework: PyTorch

Architectures: UNET, Transformer Encoder

Libraries: OpenCV, NumPy, Matplotlib, Scikit-learn

## 👨‍🎓 Author

Sivamathu Final Year Student Project Type: Academic / Final Year Project

## 📄 License

This project is for academic and research purposes only. EOF
