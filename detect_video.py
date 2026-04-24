import torch
import cv2
import numpy as np
import dlib
from torchvision import transforms
from model import create_model

# -----------------------------------------
# PERFORMANCE
# -----------------------------------------
torch.backends.cudnn.benchmark = True
cv2.setNumThreads(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------
# LOAD MODEL (UNCHANGED)
# -----------------------------------------
model = create_model(True, 0.3, pretrained=True)

checkpoint = torch.load("checkpoints/best_model.pth", map_location=device)
model.load_state_dict(checkpoint["state_dict"])

model.to(device)
model.eval()

# -----------------------------------------
# FACE DETECTOR (UNCHANGED)
# -----------------------------------------
face_detector = dlib.get_frontal_face_detector()

# -----------------------------------------
# TRANSFORM (UNCHANGED)
# -----------------------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# -----------------------------------------
# FRAME SAMPLING (IMPROVED BUT SAFE)
# -----------------------------------------
def extract_frames(video_path, max_frames=40):

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        return []

    indices = np.linspace(0, total-1, max_frames).astype(int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()
    return frames

# -----------------------------------------
# FACE CROP (EXACT SAME)
# -----------------------------------------
def get_face(frame):

    faces = face_detector(frame, 0)

    if len(faces) == 0:
        return None

    face = max(faces, key=lambda f: (f.right()-f.left())*(f.bottom()-f.top()))

    x1 = max(0, face.left())
    y1 = max(0, face.top())
    x2 = min(frame.shape[1], face.right())
    y2 = min(frame.shape[0], face.bottom())

    face_img = frame[y1:y2, x1:x2]

    if face_img.size == 0:
        return None

    return face_img

# -----------------------------------------
# LIGHTWEIGHT FEATURES (SAFE)
# -----------------------------------------
def freq_score(face):
    gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    return np.std(gray) / 255

def temporal_score(prev, curr):
    if prev is None:
        return 0
    prev = cv2.resize(prev, (64,64))
    curr = cv2.resize(curr, (64,64))
    return np.mean(cv2.absdiff(prev, curr)) / 255

# -----------------------------------------
# HEATMAP (FOR UI ONLY)
# -----------------------------------------
def generate_heatmap(face):
    gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    heat = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return cv2.addWeighted(face, 0.6, heat, 0.4, 0)

# -----------------------------------------
# MAIN DETECTION
# -----------------------------------------
def detect_video(video_path):

    frames = extract_frames(video_path)

    if len(frames) == 0:
        return "Invalid Video", 0, [], [], []

    faces = []
    temporal_scores = []
    prev_face = None

    for frame in frames:

        face = get_face(frame)

        if face is None:
            continue

        faces.append(face)
        temporal_scores.append(temporal_score(prev_face, face))
        prev_face = face

    if len(faces) < 6:
        return "No Face Detected", 0, [], [], []

    # -----------------------------------------
    # CNN INFERENCE (UNCHANGED)
    # -----------------------------------------
    batch = torch.stack([transform(f) for f in faces]).to(device)

    with torch.no_grad():
        outputs = model(batch)
        cnn_probs = torch.softmax(outputs, dim=1)[:,1].cpu().numpy()

    timeline = cnn_probs.tolist()

    # -----------------------------------------
    # CONFIDENCE FILTER (VERY IMPORTANT)
    # -----------------------------------------
    confident = cnn_probs[(cnn_probs > 0.65) | (cnn_probs < 0.35)]

    if len(confident) < 5:
        confident = cnn_probs

    # -----------------------------------------
    # FINAL SCORE (STABLE)
    # -----------------------------------------
    score = np.median(confident)

    # -----------------------------------------
    # CALIBRATION (FIX REAL WORLD)
    # -----------------------------------------
    if score > 0.5:
        score = min(score + 0.05, 1)
    else:
        score = max(score - 0.05, 0)

    confidence = round(max(score, 1-score)*100, 2)

    # -----------------------------------------
    # FINAL DECISION (ROBUST)
    # -----------------------------------------
    if score > 0.6:
        result = "DeepFake"
    elif score < 0.4:
        result = "Real"
    else:
        result = "Uncertain"

    # -----------------------------------------
    # HEATMAPS
    # -----------------------------------------
    heatmaps = [generate_heatmap(f) for f in faces[:6]]

    return result, confidence, timeline, heatmaps, []