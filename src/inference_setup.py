"""Restore the trained camera-trap model and evaluation data after a kernel restart.

This module intentionally contains inference-only setup. It never creates an
optimizer, performs backpropagation, or trains the model.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm


CLASS_NAMES = [
    "BIRD",
    "CAT",
    "EMPTY",
    "HEDGEHOG",
    "POSSUM",
    "RODENT",
]

ALERT_CLASSES = {
    "CAT",
    "HEDGEHOG",
    "POSSUM",
    "RODENT",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 1000


class CameraTrapDataset(Dataset):
    """Load cached camera-trap images in metadata row order."""

    def __init__(self, dataframe, image_dir, class_to_index, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.class_to_index = class_to_index
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_path = (
            self.image_dir
            / row["split"]
            / row["project_class"]
            / row["file"]
        )

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform:
                image = self.transform(image)

        label = self.class_to_index[row["project_class"]]
        return image, label


def _load_state_dict(checkpoint_path, device):
    try:
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def load_inference_session(
    project_root=None,
    batch_size=2,
    num_workers=0,
    device=None,
):
    """Restore notebook variables needed for validation and test inference."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]
    else:
        project_root = Path(project_root).resolve()

    processed_data_dir = project_root / "data" / "processed"
    interim_data_dir = project_root / "data" / "interim"
    cache_data_dir = project_root / "data" / "cache_1000"
    outputs_dir = project_root / "outputs"
    figures_dir = outputs_dir / "figures_v2"
    models_dir = outputs_dir / "models"
    results_dir = outputs_dir / "results_v2"
    metadata_path = interim_data_dir / "final_clean_metadata.csv"
    best_model_path = models_dir / "best_resnet18_high_res_v2.pth"

    required_paths = {
        "metadata": metadata_path,
        "image cache": cache_data_dir,
        "model checkpoint": best_model_path,
    }
    missing = [
        f"{name}: {path}"
        for name, path in required_paths.items()
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Required inference files are missing:\n" + "\n".join(missing)
        )

    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    class_names = list(CLASS_NAMES)
    class_to_index = {
        class_name: index
        for index, class_name in enumerate(class_names)
    }
    alert_indices = np.array(
        [class_to_index[name] for name in class_names if name in ALERT_CLASSES]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    df = pd.read_csv(metadata_path)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    val_dataset = CameraTrapDataset(
        val_df,
        cache_data_dir,
        class_to_index,
        transform=eval_transform,
    )
    test_dataset = CameraTrapDataset(
        test_df,
        cache_data_dir,
        class_to_index,
        transform=eval_transform,
    )

    loader_options = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    val_loader = DataLoader(val_dataset, **loader_options)
    test_loader = DataLoader(test_dataset, **loader_options)

    resnet_model = models.resnet18(weights=None)
    resnet_model.fc = nn.Linear(
        resnet_model.fc.in_features,
        len(class_names),
    )
    resnet_model.load_state_dict(
        _load_state_dict(best_model_path, device)
    )
    resnet_model = resnet_model.to(device)
    resnet_model.eval()

    return {
        "PROJECT_ROOT": project_root,
        "PROCESSED_DATA_DIR": processed_data_dir,
        "INTERIM_DATA_DIR": interim_data_dir,
        "CACHE_DATA_DIR": cache_data_dir,
        "OUTPUTS_DIR": outputs_dir,
        "FIGURES_DIR": figures_dir,
        "MODELS_DIR": models_dir,
        "RESULTS_DIR": results_dir,
        "METADATA_PATH": metadata_path,
        "BEST_MODEL_PATH": best_model_path,
        "CLASS_NAMES": class_names,
        "CLASS_TO_INDEX": class_to_index,
        "NUM_CLASSES": len(class_names),
        "ALERT_CLASSES": set(ALERT_CLASSES),
        "ALERT_INDICES": alert_indices,
        "IMAGE_SIZE": IMAGE_SIZE,
        "device": device,
        "eval_transform": eval_transform,
        "df": df,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "resnet_model": resnet_model,
        "resnet_criterion": nn.CrossEntropyLoss(),
    }


def evaluate_model_amp(model, loader, criterion, device):
    """Evaluate loss and multiclass accuracy without training."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    amp_enabled = device.type == "cuda"

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=amp_enabled)
            labels = labels.to(device, non_blocking=amp_enabled)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def run_inference(model, loader, metadata_df, device, class_names, alert_classes):
    """Return row-aligned class probabilities and alert scores."""
    class_to_index = {
        class_name: index
        for index, class_name in enumerate(class_names)
    }
    alert_indices = np.array(
        [class_to_index[name] for name in class_names if name in alert_classes]
    )
    label_batches = []
    probability_batches = []
    amp_enabled = device.type == "cuda"
    model.eval()

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Inference"):
            images = images.to(device, non_blocking=amp_enabled)
            with torch.amp.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            label_batches.append(labels.numpy())
            probability_batches.append(probabilities.cpu().numpy())

    true_indices = np.concatenate(label_batches)
    probabilities = np.concatenate(probability_batches)
    predicted_indices = probabilities.argmax(axis=1)
    results = metadata_df.reset_index(drop=True).copy()

    if len(results) != len(true_indices):
        raise ValueError("Metadata and DataLoader row counts do not match.")

    results.insert(0, "dataset_position", np.arange(len(results)))
    results["true_class_index"] = true_indices
    results["predicted_class_index"] = predicted_indices
    results["true_class"] = [class_names[index] for index in true_indices]
    results["predicted_class"] = [
        class_names[index] for index in predicted_indices
    ]

    for index, class_name in enumerate(class_names):
        results[f"prob_{class_name.lower()}"] = probabilities[:, index]

    results["true_alert"] = np.isin(true_indices, alert_indices).astype(int)
    results["argmax_alert_prediction"] = np.isin(
        predicted_indices, alert_indices
    ).astype(int)
    alert_probabilities = probabilities[:, alert_indices]
    results["sum_alert_score"] = alert_probabilities.sum(axis=1)
    results["max_alert_score"] = alert_probabilities.max(axis=1)
    return results
