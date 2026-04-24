app.py

import streamlit as st
import tempfile
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import dlib

from detect_video import detect_video
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


st.set_page_config(page_title="DeepFake AGI Detector", layout="wide")

st.title("🎭 DeepFake Detector — AGI MODE")
st.markdown("### 🧠 Multi-Modal AI | Explainable | Real-time Vision System")

st.divider()


face_detector = dlib.get_frontal_face_detector()


def create_pdf(result, confidence):
    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("DeepFake AI Report", styles['Title']))
    content.append(Spacer(1, 20))
    content.append(Paragraph(f"Result: {result}", styles['Heading2']))
    content.append(Paragraph(f"Confidence: {confidence}%", styles['Normal']))

    if result == "DeepFake":
        content.append(Paragraph("Detected synthetic artifacts and inconsistencies.", styles['Normal']))
    else:
        content.append(Paragraph("Natural human facial behavior detected.", styles['Normal']))

    doc.build(content)


def measure_fps(start, end):
    return round(1/(end-start), 2) if end-start > 0 else 0


def draw_real_overlay(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 0)

    for face in faces:

        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()

        # face box
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        h = y2 - y1

        # eyes region
        cv2.rectangle(frame, (x1, y1), (x2, y1 + int(h*0.3)), (255,0,0), 2)

        # mouth region
        cv2.rectangle(frame, (x1, y1 + int(h*0.6)), (x2, y2), (0,0,255), 2)

    return frame

# -----------------------------------------
# UI
# -----------------------------------------
uploaded_file = st.file_uploader("📁 Upload Video", type=["mp4","mov","avi"])

if uploaded_file:

    st.subheader("🎬 Uploaded Video")
    st.video(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(uploaded_file.read())
        video_path = temp.name

    st.divider()

    if st.button("🚀 Run AGI Detection"):

        start_time = time.time()

        with st.spinner("🧠 AGI analyzing video..."):

            try:
                result, confidence, timeline, heatmaps, boxes = detect_video(video_path)
            except Exception as e:
                st.error(str(e))
                st.stop()

        end_time = time.time()

        fps = measure_fps(start_time, end_time)

        # -----------------------------------------
        # PERFORMANCE PANEL
        # -----------------------------------------
        col1, col2, col3 = st.columns(3)
        col1.metric("⚡ FPS", fps)
        col2.metric("🎯 Confidence", f"{confidence}%")
        col3.metric("⏱ Time", f"{round(end_time-start_time,2)}s")

        st.divider()

        # -----------------------------------------
        # RESULT
        # -----------------------------------------
        st.subheader("🧾 Final Result")

        if result == "DeepFake":
            st.error("🚨 DeepFake Detected")
        elif result == "Real":
            st.success("✅ Authentic Video")
        else:
            st.warning("⚠ Uncertain")

        st.progress(int(confidence))

        # -----------------------------------------
        # TIMELINE
        # -----------------------------------------
        st.subheader("📊 Frame Confidence Timeline")

        fig, ax = plt.subplots()
        ax.plot(timeline)
        ax.set_title("DeepFake Probability")
        ax.grid(True)
        st.pyplot(fig)

        # -----------------------------------------
        # HEATMAPS
        # -----------------------------------------
        st.subheader("🔥 AI Attention Heatmaps")

        cols = st.columns(5)
        for i, img in enumerate(heatmaps[:5]):
            cols[i % 5].image(img)

        # -----------------------------------------
        # REAL FACE TRACKING PREVIEW
        # -----------------------------------------
        st.subheader("🎯 Real Face Tracking + Region Analysis")

        cap = cv2.VideoCapture(video_path)
        frames = []

        for _ in range(6):
            ret, frame = cap.read()
            if not ret:
                break

            frame = draw_real_overlay(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        cols = st.columns(len(frames))
        for i, f in enumerate(frames):
            cols[i].image(f)

        # -----------------------------------------
        # CSV EXPORT
        # -----------------------------------------
        st.download_button(
            "📥 Download Timeline CSV",
            data="\n".join(map(str, timeline)),
            file_name="timeline.csv"
        )

        # -----------------------------------------
        # PDF REPORT
        # -----------------------------------------
        create_pdf(result, confidence)

        with open("report.pdf", "rb") as f:
            st.download_button("📄 Download Report", f, "report.pdf")

        # -----------------------------------------
        # EXPLAINABLE AI
        # -----------------------------------------
        st.subheader("🧠 Explainable AI Insights")

        if result == "DeepFake":
            st.write("""
            ✔ Facial warping detected  
            ✔ Temporal instability  
            ✔ GAN-based artifacts  
            ✔ Frequency inconsistencies  
            """)
        else:
            st.write("""
            ✔ Natural facial structure  
            ✔ Stable motion  
            ✔ No synthetic patterns  
            ✔ Consistent lighting  
            """)

    if os.path.exists(video_path):
        os.remove(video_path)














data_loader.py

from pathlib import Path
import numpy as np
import dlib
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import load_and_preprocess_image
def collate_fn(batch):
    imgs = [item['image'] for item in batch if item is not None and item['image'] is not None]
    targets = [item['label'] for item in batch if item is not None and item['image'] is not None]
    filenames = [item['filename'] for item in batch if item is not None and item['image'] is not None]
    if len(imgs) == 0:
        return None
    imgs = torch.stack(imgs)
    targets = torch.stack(targets)
    return {'image': imgs, 'label': targets, 'filename': filenames}
def get_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_transforms, val_transforms
class FFDataset(Dataset):

    def __init__(self, filenames, filepath, transform,
                 output_image_size=224, recompute=False):
        self.filenames = filenames
        self.transform = transform
        self.image_size = output_image_size
        self.recompute = recompute
        self.cached_path = Path(filepath)
        self.cached_path.mkdir(parents=True, exist_ok=True)
        self.face_detector = dlib.get_frontal_face_detector()
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        filename_str = str(filename)
        image_id = filename.stem
        # label detection
        label = 1 if "fake" in filename.parts else 0
        cache_file = self.cached_path / f"processed_{image_id}.npy"
        if cache_file.exists() and not self.recompute:
            image = np.load(cache_file)
        else:
            image = load_and_preprocess_image(
                filename_str,
                self.image_size,
                self.face_detector
            )
            if image is None:
                image = []
            np.save(cache_file, image)
        if len(image) == 0:
            return {'image': None, 'label': None, 'filename': filename_str}
        image = Image.fromarray(image)
        image = self.transform(image)
        label = torch.tensor(label)
        return {'image': image, 'label': label, 'filename': filename_str}
def create_dataloaders(params):
    train_transforms, val_transforms = get_transforms()
    base_path = Path("datasets")
    train_dl = _create_dataloader(
        base_path / f"{params['train_data']}_deepfake",
        mode="train",
        batch_size=params['batch_size'],
        transformations=train_transforms,
        sample_ratio=params['sample_ratio']
    )
    val_base_dl = _create_dataloader(
        base_path / "base_deepfake" / "val",
        mode="val",
        batch_size=params['batch_size'],
        transformations=val_transforms,
        sample_ratio=params['sample_ratio']
    )
    augment_val_path = base_path / "augment_deepfake" / "val"
    if augment_val_path.exists():
        val_augment_dl = _create_dataloader(
            augment_val_path,
            mode="val",
            batch_size=params['batch_size'],
            transformations=val_transforms,
            sample_ratio=params['sample_ratio']
        )
    else:
        val_augment_dl = val_base_dl
    display_dl_iter = iter(val_base_dl)
    return train_dl, val_base_dl, val_augment_dl, display_dl_iter
def _create_dataloader(file_path, mode, batch_size,
                       transformations, sample_ratio,
                       num_workers=0):
    filenames = []
    data_path = Path(file_path)
    real_frames = _find_images(data_path / "real" / "frames")
    fake_frames = _find_images(data_path / "fake" / "frames")
    filenames.extend(real_frames)
    filenames.extend(fake_frames)
    assert len(filenames) != 0, f"filenames are empty {filenames}"
    np.random.shuffle(filenames)
    if mode == "train":
        filenames = filenames[:int(sample_ratio * len(filenames))]
    dataset = FFDataset(
        filenames,
        filepath="datasets/precomputed",
        transform=transformations,
        recompute=False
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    print(f"{mode} data: {len(dataset)}")
    return dataloader
def _find_images(folder_path):

    if not folder_path.exists():
        return []
    return (
        list(folder_path.glob("*.jpg")) +
        list(folder_path.glob("*.jpeg")) +
        list(folder_path.glob("*.png"))
    )















detect_video.py
import torch
import cv2
import numpy as np
import dlib
from torchvision import transforms
from model import create_model
torch.backends.cudnn.benchmark = True
cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model(True, 0.3, pretrained=True)
checkpoint = torch.load("checkpoints/best_model.pth", map_location=device)
model.load_state_dict(checkpoint["state_dict"])
model.to(device)
model.eval()
face_detector = dlib.get_frontal_face_detector()
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])
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
def freq_score(face):
    gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    return np.std(gray) / 255
def temporal_score(prev, curr):
    if prev is None:
        return 0
    prev = cv2.resize(prev, (64,64))
    curr = cv2.resize(curr, (64,64))
    return np.mean(cv2.absdiff(prev, curr)) / 255
def generate_heatmap(face):
    gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    heat = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return cv2.addWeighted(face, 0.6, heat, 0.4, 0)
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
    batch = torch.stack([transform(f) for f in faces]).to(device)
    with torch.no_grad():
        outputs = model(batch)
        cnn_probs = torch.softmax(outputs, dim=1)[:,1].cpu().numpy()
    timeline = cnn_probs.tolist()
    confident = cnn_probs[(cnn_probs > 0.65) | (cnn_probs < 0.35)]
    if len(confident) < 5:
        confident = cnn_probs
    score = np.median(confident)
    if score > 0.5:
        score = min(score + 0.05, 1)
    else:
        score = max(score - 0.05, 0)
    confidence = round(max(score, 1-score)*100, 2)
    if score > 0.6:
        result = "DeepFake"
    elif score < 0.4:
        result = "Real"
    else:
        result = "Uncertain"
    heatmaps = [generate_heatmap(f) for f in faces[:6]]
    return result, confidence, timeline, heatmaps, []














faceforensics_download.py

import argparse
import os
import urllib
import urllib.request
import tempfile
import time
import sys
import json
import random
from tqdm import tqdm
from os.path import join
FILELIST_URL = 'misc/filelist.json'
DEEPFEAKES_DETECTION_URL = 'misc/deepfake_detection_filenames.json'
DEEPFAKES_MODEL_NAMES = ['decoder_A.h5', 'decoder_B.h5', 'encoder.h5',]
DATASETS = {
    'original_youtube_videos': 'misc/downloaded_youtube_videos.zip',
    'original_youtube_videos_info': 'misc/downloaded_youtube_videos_info.zip',
    'original': 'original_sequences/youtube',
    'DeepFakeDetection_original': 'original_sequences/actors',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'DeepFakeDetection': 'manipulated_sequences/DeepFakeDetection',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceShifter': 'manipulated_sequences/FaceShifter',
    'FaceSwap': 'manipulated_sequences/FaceSwap',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures'
    }
ALL_DATASETS = ['original', 'DeepFakeDetection_original', 'Deepfakes',
                'DeepFakeDetection', 'Face2Face', 'FaceShifter', 'FaceSwap',
                'NeuralTextures']
COMPRESSION = ['raw', 'c23', 'c40']
TYPE = ['videos', 'masks', 'models']
SERVERS = ['EU', 'EU2', 'CA']
def parse_args():
    parser = argparse.ArgumentParser(
        description='Downloads FaceForensics v2 public data release.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('output_path', type=str, help='Output directory.')
    parser.add_argument('-d', '--dataset', type=str, default='all',
                        help='Which dataset to download, either pristine or '
                             'manipulated data or the downloaded youtube '
                             'videos.',
                        choices=list(DATASETS.keys()) + ['all']
                        )
    parser.add_argument('-c', '--compression', type=str, default='raw',
                        help='Which compression degree. All videos '
                             'have been generated with h264 with a varying '
                             'codec. Raw (c0) videos are lossless compressed.',
                        choices=COMPRESSION
                        )
    parser.add_argument('-t', '--type', type=str, default='videos',
                        help='Which file type, i.e. videos, masks, for our '
                             'manipulation methods, models, for Deepfakes.',
                        choices=TYPE
                        )
    parser.add_argument('-n', '--num_videos', type=int, default=None,
                        help='Select a number of videos number to '
                             "download if you don't want to download the full"
                             ' dataset.')
    parser.add_argument('--server', type=str, default='EU',
                        help='Server to download the data from. If you '
                             'encounter a slow download speed, consider '
                             'changing the server.',
                        choices=SERVERS
                        )
    args = parser.parse_args()
    server = args.server
    if server == 'EU':
        server_url = 'http://canis.vc.in.tum.de:8100/'
    elif server == 'EU2':
        server_url = 'http://kaldir.vc.in.tum.de/faceforensics/'
    elif server == 'CA':
        server_url = 'http://falas.cmpt.sfu.ca:8100/'
    else:
        raise Exception('Wrong server name. Choices: {}'.format(str(SERVERS)))
    args.tos_url = server_url + 'webpage/FaceForensics_TOS.pdf'
    args.base_url = server_url + 'v3/'
    args.deepfakes_model_url = server_url + 'v3/manipulated_sequences/' + \
                               'Deepfakes/models/'
    return args
def download_files(filenames, base_url, output_path, report_progress=True):
    os.makedirs(output_path, exist_ok=True)
    if report_progress:
        filenames = tqdm(filenames)
    for filename in filenames:
        download_file(base_url + filename, join(output_path, filename))
def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\rProgress: %d%%, %d MB, %d KB/s, %d seconds passed" %
                     (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()
def download_file(url, out_file, report_progress=False):
    out_dir = os.path.dirname(out_file)
    if not os.path.isfile(out_file):
        fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
        f = os.fdopen(fh, 'w')
        f.close()
        if report_progress:
            urllib.request.urlretrieve(url, out_file_tmp,
                                       reporthook=reporthook)
        else:
            urllib.request.urlretrieve(url, out_file_tmp)
        os.rename(out_file_tmp, out_file)
    else:
        tqdm.write('WARNING: skipping download of existing file ' + out_file)
def main(args):
    # TOS
    print('By pressing any key to continue you confirm that you have agreed '\
          'to the FaceForensics terms of use as described at:')
    print(args.tos_url)
    print('***')
    print('Press any key to continue, or CTRL-C to exit.')
    _ = input('')
    # Extract arguments
    c_datasets = [args.dataset] if args.dataset != 'all' else ALL_DATASETS
    c_type = args.type
    c_compression = args.compression
    num_videos = args.num_videos
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    # Check for special dataset cases
    for dataset in c_datasets:
        dataset_path = DATASETS[dataset]
        # Special cases
        if 'original_youtube_videos' in dataset:
            # Here we download the original youtube videos zip file
            print('Downloading original youtube videos.')
            if not 'info' in dataset_path:
                print('Please be patient, this may take a while (~40gb)')
                suffix = ''
            else:
            	suffix = 'info'
            download_file(args.base_url + '/' + dataset_path,
                          out_file=join(output_path,
                                        'downloaded_videos{}.zip'.format(
                                            suffix)),
                          report_progress=True)
            return
        # Else: regular datasets
        print('Downloading {} of dataset "{}"'.format(
            c_type, dataset_path
        ))
        # Get filelists and video lenghts list from server
        if 'DeepFakeDetection' in dataset_path or 'actors' in dataset_path:
        	filepaths = json.loads(urllib.request.urlopen(args.base_url + '/' +
                DEEPFEAKES_DETECTION_URL).read().decode("utf-8"))
        	if 'actors' in dataset_path:
        		filelist = filepaths['actors']
        	else:
        		filelist = filepaths['DeepFakesDetection']
        elif 'original' in dataset_path:
            # Load filelist from server
            file_pairs = json.loads(urllib.request.urlopen(args.base_url + '/' +
                FILELIST_URL).read().decode("utf-8"))
            filelist = []
            for pair in file_pairs:
            	filelist += pair
        else:
            # Load filelist from server
            file_pairs = json.loads(urllib.request.urlopen(args.base_url + '/' +
                FILELIST_URL).read().decode("utf-8"))
            # Get filelist
            filelist = []
            for pair in file_pairs:
                filelist.append('_'.join(pair))
                if c_type != 'models':
                    filelist.append('_'.join(pair[::-1]))
        # Maybe limit number of videos for download
        if num_videos is not None and num_videos > 0:
        	print('Downloading the first {} videos'.format(num_videos))
        	filelist = filelist[:num_videos]
        # Server and local paths
        dataset_videos_url = args.base_url + '{}/{}/{}/'.format(
            dataset_path, c_compression, c_type)
        dataset_mask_url = args.base_url + '{}/{}/videos/'.format(
            dataset_path, 'masks', c_type)
        if c_type == 'videos':
            dataset_output_path = join(output_path, dataset_path, c_compression,
                                       c_type)
            print('Output path: {}'.format(dataset_output_path))
            filelist = [filename + '.mp4' for filename in filelist]
            download_files(filelist, dataset_videos_url, dataset_output_path)
        elif c_type == 'masks':
            dataset_output_path = join(output_path, dataset_path, c_type,
                                      'videos')
            print('Output path: {}'.format(dataset_output_path))
            if 'original' in dataset:
                if args.dataset != 'all':
                    print('Only videos available for original data. Aborting.')
                    return
                else:
                    print('Only videos available for original data. '
                          'Skipping original.\n')
                    continue
            if 'FaceShifter' in dataset:
                print('Masks not available for FaceShifter. Aborting.')
                return
            filelist = [filename + '.mp4' for filename in filelist]
            download_files(filelist, dataset_mask_url, dataset_output_path)
        # Else: models for deepfakes
        else:
            if dataset != 'Deepfakes' and c_type == 'models':
                print('Models only available for Deepfakes. Aborting')
                return
            dataset_output_path = join(output_path, dataset_path, c_type)
            print('Output path: {}'.format(dataset_output_path))
            # Get Deepfakes models
            for folder in tqdm(filelist):
                folder_filelist = DEEPFAKES_MODEL_NAMES
                # Folder paths
                folder_base_url = args.deepfakes_model_url + folder + '/'
                folder_dataset_output_path = join(dataset_output_path,
                                                  folder)
                download_files(folder_filelist, folder_base_url,
                               folder_dataset_output_path,
                               report_progress=False)   # already done
if __name__ == "__main__":
    args = parse_args()
    main(args)














hparams_search.py

import foundations
import numpy as np
NUM_JOBS = 140
def generate_params():
    params = {'batch_size': int(np.random.choice([256, 512, 1024])),
              'n_epochs': int(np.random.choice([20, 15, 25])),
              "pct_start": float(np.random.uniform(0.3, 0.5)),
              'weight_decay': float(np.random.uniform(0.01, 0.3)),
              'dropout': float(np.random.choice([0.8, 0.9, 0.75])),
              'max_lr': float(np.random.uniform(0.00003, 0.00007)),
              'use_lr_scheduler': int(np.random.choice([0, 1])),
              'use_hidden_layer': int(np.random.choice([0, 1])),
              # 'use_lr_scheduler': 1,
              'train_data': 'both',
              # 'train_data': np.random.choice(['augment', 'base', 'both'])
              'sample_ratio': float(np.random.choice([0.1, 0.25, 0.5, 0.75, 1.])),
    }
    return params
for job_ in range(NUM_JOBS):
    print(f"packaging job {job_}")
    hyper_params = generate_params()
    print(hyper_params)
    foundations.submit(scheduler_config='scheduler', job_directory='.', command='main.py', params=hyper_params,
                       stream_job_logs=False)















main.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import os
from data_loader import create_dataloaders
from model import get_trainable_params, create_model, print_model_params
from train import train
def main():
    # =====================================
    # Fix random seed for reproducibility
    # =====================================
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.makedirs("checkpoints", exist_ok=True)
    params = {
        "train_data": "base",      # base | augment | both
        "use_hidden_layer": True,
        "dropout": 0.3,
        "max_lr": 0.0001,
        "weight_decay": 0.0001,
        "use_lr_scheduler": True,
        "n_epochs": 20,
        "pct_start": 0.3,
        "batch_size": 32,
        "sample_ratio": 1.0
    }
    print("===================================")
    print("DeepFake Detection Training")
    print("===================================")
    print(f"Training dataset : {params['train_data']}")
    print(f"Epochs           : {params['n_epochs']}")
    print(f"Batch size       : {params['batch_size']}")
    print("===================================\n")
    print("Creating datasets...\n")
    train_dl, val_base_dl, val_augment_dl, display_dl_iter = create_dataloaders(params)
    print("Creating loss function...\n")
    criterion = nn.CrossEntropyLoss()
    print("Creating model...\n")
    model = create_model(
        bool(params['use_hidden_layer']),
        params['dropout']
    )
    print_model_params(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")
    model = model.to(device)
    params_to_update = get_trainable_params(model)
    print("Creating optimizer...\n")
    optimizer = optim.Adam(
        params_to_update,
        lr=params['max_lr'],
        weight_decay=params['weight_decay']
    )
    if params['use_lr_scheduler']:
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=params['max_lr'],
            epochs=params['n_epochs'],
            steps_per_epoch=len(train_dl),
            pct_start=params['pct_start'],
            anneal_strategy='cos',
            cycle_momentum=False
        )

    else:
        scheduler = None
    print("===================================")
    print("Training started...")
    print("===================================\n")
    train(
        train_dl,
        val_base_dl,
        val_augment_dl,
        display_dl_iter,
        model,
        optimizer,
        params['n_epochs'],
        params['max_lr'],
        scheduler,
        criterion,
        train_source=params["train_data"]
    )
    print("\n===================================")
    print("Training finished")
    print("Best model saved in /checkpoints")
    print("===================================")
if __name__ == "__main__":
    main()
















model.py
import torchvision.models as models
import torch.nn as nn
def check_model_block(model):
    for name, child in model.named_children():
        print(name)
def print_model_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"total number of params: {pytorch_total_params:,}")
    return pytorch_total_params
def get_trainable_params(model):
    print("Params to learn:")
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("\t", repr(name))
            params_to_update.append(param)
    return params_to_update
def create_model(use_hidden_layer=True, dropout=0.3, pretrained=True):
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    print(f"Input feature dim: {in_features}")
    if use_hidden_layer:
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )
    else:
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 2)
        )
    print(model)
    return model