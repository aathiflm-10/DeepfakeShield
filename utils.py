import cv2
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# =====================================================
# Improved Bounding Box
# =====================================================
def get_boundingbox(face, width, height, scale=1.5, minsize=None):

    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()

    size_bb = int(max(x2 - x1, y2 - y1) * scale)

    if minsize and size_bb < minsize:
        size_bb = minsize

    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)

    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


# =====================================================
# Image preprocessing
# =====================================================
def load_and_preprocess_image(image_filename, output_image_size, face_detector):

    image = cv2.imread(image_filename)

    if image is None:
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cropped_image = get_face_crop(face_detector, image)

    if cropped_image is None:
        return None

    resized_image = cv2.resize(
        cropped_image,
        (output_image_size, output_image_size)
    )

    return resized_image


# =====================================================
# Better face cropping
# =====================================================
def get_face_crop(face_detector, image):

    faces = face_detector(image, 1)

    height, width = image.shape[:2]

    if len(faces) == 0:
        return None

    # Choose the biggest face
    biggest_face = None
    biggest_area = 0

    for face in faces:

        area = (face.right() - face.left()) * (face.bottom() - face.top())

        if area > biggest_area:
            biggest_area = area
            biggest_face = face

    x, y, size = get_boundingbox(biggest_face, width, height)

    cropped_face = image[y:y + size, x:x + size]

    if cropped_face.size == 0:
        return None

    return cropped_face


# =====================================================
# Metrics Visualization
# =====================================================
def visualize_metrics(records, extra_metric, name):

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 6))

    axes[0].plot(records.train_losses, label='train')
    axes[0].plot(records.train_losses_wo_dropout, label='train w/o dropout')
    axes[0].plot(records.base_val_losses, label='base_val')
    axes[0].plot(records.augment_val_losses, label='augment_val')
    axes[0].set_title('Loss')
    axes[0].legend()

    axes[1].plot(records.train_accs, label='train')
    axes[1].plot(records.train_accs_wo_dropout, label='train w/o dropout')
    axes[1].plot(records.base_val_accs, label='base_val')
    axes[1].plot(records.augment_val_accs, label='augment_val')
    axes[1].axhline(y=0.5, color='g', linestyle='--')
    axes[1].set_title('Accuracy')
    axes[1].legend()

    axes[2].plot(records.train_custom_metrics, label='train')
    axes[2].plot(records.train_custom_metrics_wo_dropout, label='train w/o dropout')
    axes[2].plot(records.base_val_custom_metrics, label='base_val')
    axes[2].plot(records.augment_val_custom_metrics, label='augment_val')
    axes[2].set_title(extra_metric.__name__)
    axes[2].legend()

    axes[3].plot(records.lrs)
    axes[3].set_title('Learning Rate')

    plt.tight_layout()
    plt.savefig(name, format='png')
    plt.close()


# =====================================================
# Display Predictions
# =====================================================
def display_predictions_on_image(model, precomputed_cached_path, val_iter, name):

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        data = next(val_iter)
    except StopIteration:
        return

    inputs = data['image'].to(device)
    labels = data['label'].view(-1).to(device)
    filenames = data['filename']

    with torch.no_grad():
        outputs = model(inputs)
        outputs_probability = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

    n_images = len(inputs)
    nrows = int(np.sqrt(n_images))
    ncols = int(np.ceil(n_images / nrows))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20))
    axes = np.array(axes).reshape(-1)

    for idx in range(n_images):

        image_id = Path(filenames[idx]).stem
        face_crop_path = precomputed_cached_path / f'processed_{image_id}.npy'

        if face_crop_path.exists():
            face_crop = np.load(face_crop_path)
            axes[idx].imshow(face_crop)
        else:
            axes[idx].imshow(np.zeros((224,224,3)))

        axes[idx].set_title(
            f'{outputs_probability[idx][0]:.2f},'
            f'{outputs_probability[idx][1]:.2f} | '
            f'Pred:{predicted[idx].item()} '
            f'True:{labels[idx].item()}'
        )

        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(name, format='png')
    plt.close()


# =====================================================
# Parameter Parsing
# =====================================================
def parse_and_override_params(params):

    data_dict = {'base': 0, 'augment': 1, 'both': 2}

    parsed_params = params.copy()

    if isinstance(params['train_data'], str):
        parsed_params['train_data'] = data_dict[params['train_data']]

    return data_dict