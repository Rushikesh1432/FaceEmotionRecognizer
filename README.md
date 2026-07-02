# Face Emotion Recognizer

A Convolutional Neural Network (CNN) based system for classifying human facial emotions from images, with real-time detection via webcam. Built as an end-to-end deep learning project — from data pipeline to a live demo.

---

## 📌 Overview

This project classifies facial expressions into **7 emotion categories**:
`Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, `Neutral`

It uses the **FER-2013** dataset and a custom CNN architecture trained from scratch, with plans to compare against a transfer-learning baseline.

---

## 🎯 Features

- Custom CNN built with TensorFlow/Keras
- Modular, reusable project structure (data / model / training separated)
- Real-time emotion detection via webcam (OpenCV face detection + CNN inference)
- Training visualization (loss/accuracy curves)
- Grad-CAM interpretability (visualizes which facial regions drive predictions)

---

## 🗂️ Project Structure

```
FaceEmotionRecognizer/
├── data/
│   └── fer2013/
│       ├── train/          # training images, one folder per emotion class
│       └── test/           # test images, one folder per emotion class
├── src/
│   ├── data/
│   │   └── dataset.py       # data loading & augmentation pipeline
│   ├── models/
│   │   └── cnn_baseline.py  # CNN architecture definition
│   └── training/
│       └── train.py         # training loop with checkpointing
├── notebooks/
│   ├── eda.ipynb            # exploratory data analysis
│   └── train.ipynb          # interactive training/experimentation
├── app/
│   └── webcam_demo.py       # real-time webcam emotion detection
├── main.py                  # end-to-end pipeline entry point
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

**FER-2013** — 35,887 grayscale, 48×48 pixel facial images across 7 emotion classes.

- Source: [Kaggle — FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)
- Note: FER-2013 is a known benchmark with inherent label ambiguity; even published models typically reach 65–75% accuracy, and human-level accuracy on this dataset is estimated around 65%.

After downloading, place the dataset inside `data/fer2013/` following the `train/` and `test/` folder structure shown above (each containing one subfolder per emotion).

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/Rushikesh1432/FaceEmotionRecognizer.git
cd FaceEmotionRecognizer
```

### 2. Create and activate a conda environment
```bash
conda create -n emotion-rec python=3.10 -y
conda activate emotion-rec
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Download FER-2013 from Kaggle and place it under `data/fer2013/` as described above.

---

## 🚀 Usage

### Train the model
```bash
python main.py
```

### Explore data / experiment interactively
Open `notebooks/eda.ipynb` or `notebooks/train.ipynb` in Jupyter.

### Run the real-time webcam demo
```bash
python app/webcam_demo.py
```

---

## 🧠 Model Architecture

Custom CNN:

| Layer | Details |
|---|---|
| Input | 48×48×1 (grayscale) |
| Conv Block 1 | Conv2D(32) → MaxPool |
| Conv Block 2 | Conv2D(64) → MaxPool |
| Conv Block 3 | Conv2D(128) → MaxPool |
| Dense | 64 → 32 |
| Output | Dense(7, softmax) |

**Loss:** Categorical Cross-Entropy
**Optimizer:** Adam

---

## 📈 Results

> _To be updated after training._

| Metric | Value |
|---|---|
| Train Accuracy | `TBD` |
| Validation Accuracy | `TBD` |
| Test Accuracy | `TBD` |
| Test Loss | `TBD` |

**Per-class performance (F1-score):**

| Emotion | F1-score |
|---|---|
| Angry | `TBD` |
| Disgust | `TBD` |
| Fear | `TBD` |
| Happy | `TBD` |
| Sad | `TBD` |
| Surprise | `TBD` |
| Neutral | `TBD` |

**Confusion Matrix:**

> _Insert confusion matrix image here after training._

**Training curves:**

> _Insert loss/accuracy plot here after training._

---

## 🎥 Demo

> _Insert webcam demo screenshot or GIF here._

---

## 🔍 Interpretability (Grad-CAM)

> _Insert Grad-CAM visualization examples here — shows which facial regions the model focuses on for each emotion prediction._

---

## 🛠️ Tech Stack

- Python 3.10
- TensorFlow / Keras
- OpenCV
- NumPy, Pandas, Matplotlib, Scikit-learn

---

## 📌 Future Improvements

- Compare custom CNN against transfer learning (MobileNetV2 / ResNet50)
- Address class imbalance (class weighting / focal loss)
- Deploy as a web app (Streamlit / Gradio)
- Expand to video-based temporal emotion tracking

---

## 👤 Author

**Rushikesh**
CSE (AI/ML), Keshav Memorial Institute of Technology

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
