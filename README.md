# 🎭 DeepFake Detection Shield — AGI Mode

A professional-grade DeepFake detection system using Multi-Modal AI and Explainable Vision systems. This project can detect synthetic facial warping, temporal instabilities, and GAN-based artifacts in videos.

## 🚀 Key Features

### 1. Streamlit Web Dashboard (`app.py`)
- **Video Upload**: Supports `.mp4`, `.avi`, and `.mov`.
- **AGI Detection Mode**: analyzes frames in real-time.
- **Explainable AI (XAI)**: Provides insights on why a video was flagged (e.g., "GAN artifacts detected").
- **AI Attention Heatmaps**: Visualizes which parts of the face the AI is focusing on.
- **Reporting**: Generates professional PDF reports and CSV timelines for forensic analysis.

### 2. Detection Engine (`detect_video.py`)
- **Face Tracking**: Uses `dlib` for robust frontal face detection.
- **Confidence Calibration**: Uses median filtering across frames to provide a stable, reliable score.
- **Temporal Analysis**: Checks for inconsistencies between consecutive frames (a common weakness in deepfakes).

### 3. Model Architecture (`model.py`)
- **Backbone**: ResNet18 (Pre-trained on ImageNet).
- **Custom Classifier**: A multi-layer head with Dropout and Batch Normalization to prevent overfitting.
- **Accuracy**: Optimized for generalizeable detection across both high-quality and low-quality manipulations.

---

## 🛠 Technology Stack

- **Logic**: Python 3.10
- **Deep Learning**: PyTorch, Torchvision
- **Computer Vision**: OpenCV, Dlib
- **Web UI**: Streamlit
- **Reporting**: ReportLab

---

## 📂 Project Structure

- `app.py`: The main web application interface.
- `detect_video.py`: The core logic for running the AI on a video file.
- `main.py` & `train.py`: The pipeline for training the AI model on new datasets.
- `model.py`: Defines the ResNet18-based architecture.
- `data_loader.py`: Handles complex image transformations and batching.
- `checkpoints/`: Stores the trained model weights (`best_model.pth`).
- `datasets/`: (Local only) Folder containing training data (121 GB).

---

## 💻 Usage

### Running the Web App
1. Open your terminal in the project folder.
2. Run the following command:
   ```bash
   streamlit run app.py
   ```
3. Upload a video and click **"Run AGI Detection"**.

### Training the Model
If you want to re-train the model on your own data:
```bash
python main.py
```

---

## ⚠️ Important Note on Data
The **121 GB** dataset (including `datasets/` and `checkpoints/`) is too large for GitHub and is currently **not uploaded** to the online repository. 

**Recommendation**: Store your datasets on a cloud service (Google Drive, AWS, or Kaggle) and keep only the code on GitHub.

---

## 📜 License
© 2026 DeepfakeShield. Built for Final Year Project - DeepFake Detection.