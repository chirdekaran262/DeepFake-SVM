# 🎧 DeepFake Audio Detection using SVM

Detect whether an audio file is real or fake using MFCC features and a Support Vector Machine (SVM) classifier.

## 🚀 Features

- Extracts MFCC features from audio
- Uses SVM for classification
- Web interface via Flask for file upload and prediction

## 🗂️ Project Structure

```
DeepFake-SVM/
├── app.py               # Flask app
├── pe_classifier.py     # Prediction logic
├── templates/
│   └── index.html       # Frontend UI
├── models/              # Saved model
├── features/            # Extracted features
├── dataset/             # Audio files
├── requirements.txt
└── README.md
├── real/          # Directory containing real audio files
└── fake/          # Directory containing fake audio files
```

## 🎙️ Dataset Setup

Download your dataset and organize it in the following structure in the project root:

```
DeepFake-SVM/
├── real/          # Directory containing real audio files
└── fake/          # Directory containing fake audio files
```

Place all **real** audio samples in the `real/` folder and all **fake** audio samples in the `fake/` folder.## 🛠️ Setup Instructions
download from kaggle 

### 1. Clone the Repository

```bash
git clone https://github.com/chirdekaran262/DeepFake-SVM.git
cd DeepFake-SVM
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
python app.py
```

Then open your browser and go to: [http://localhost:5000](http://localhost:5000)

## 📄 License

This project is licensed under the MIT License.

