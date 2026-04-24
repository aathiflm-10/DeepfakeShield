from pathlib import Path
import numpy as np
import dlib
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils import load_and_preprocess_image


# =====================================================
# Collate Function
# =====================================================
def collate_fn(batch):

    imgs = [item['image'] for item in batch if item is not None and item['image'] is not None]
    targets = [item['label'] for item in batch if item is not None and item['image'] is not None]
    filenames = [item['filename'] for item in batch if item is not None and item['image'] is not None]

    if len(imgs) == 0:
        return None

    imgs = torch.stack(imgs)
    targets = torch.stack(targets)

    return {'image': imgs, 'label': targets, 'filename': filenames}


# =====================================================
# Image Transforms
# =====================================================
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


# =====================================================
# Dataset Class
# =====================================================
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


# =====================================================
# Create Dataloaders
# =====================================================
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


# =====================================================
# Internal DataLoader Creator
# =====================================================
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


# =====================================================
# Image Finder
# =====================================================
def _find_images(folder_path):

    if not folder_path.exists():
        return []

    return (
        list(folder_path.glob("*.jpg")) +
        list(folder_path.glob("*.jpeg")) +
        list(folder_path.glob("*.png"))
    )