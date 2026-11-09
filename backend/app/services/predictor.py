import base64
import io
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "steel_defect_model.pth"
DEVICE = torch.device("cpu")

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

PATCH_SIZE = 200
PATCH_STRIDE = 100
INFERENCE_BATCH_SIZE = 32
MAX_PATCHES = 5000
MAX_GRADCAM_EXPLANATIONS = 6

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

_model = None


def load_model():
    global _model
    if _model is not None:
        return _model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {MODEL_PATH}. "
            "Copy steel_defect_model.pth into the backend folder."
        )

    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        saved_classes = checkpoint.get("class_names", CLASS_NAMES)
    else:
        state_dict = checkpoint
        saved_classes = CLASS_NAMES

    if list(saved_classes) != CLASS_NAMES:
        raise ValueError(
            f"Class order mismatch. Checkpoint classes: {saved_classes}; "
            f"application classes: {CLASS_NAMES}"
        )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    _model = model
    return _model


def _to_data_url(rgb_array: np.ndarray, image_format="PNG") -> str:
    image = Image.fromarray(np.uint8(np.clip(rgb_array, 0, 255)))
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{image_format.lower()};base64,{encoded}"


def _axis_starts(length: int, patch_size: int, stride: int):
    """Return patch starts and always include the far edge of the image."""
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    last_start = length - patch_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _extract_patches(image: Image.Image, patch_size=PATCH_SIZE, stride=PATCH_STRIDE):
    width, height = image.size
    xs = _axis_starts(width, patch_size, stride)
    ys = _axis_starts(height, patch_size, stride)

    patches = []
    for y in ys:
        for x in xs:
            right = min(x + patch_size, width)
            bottom = min(y + patch_size, height)
            patch = image.crop((x, y, right, bottom))
            patches.append({
                "image": patch,
                "bbox": [x, y, right, bottom],
            })

    if len(patches) > MAX_PATCHES:
        raise ValueError(
            f"This image creates {len(patches)} patches, exceeding the "
            f"current limit of {MAX_PATCHES}. Use a smaller image or increase "
            "MAX_PATCHES after performance testing."
        )
    return patches


def _box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _merge_overlapping_predictions(predictions, iou_threshold=0.25):
    """Greedy class-wise NMS to reduce duplicate overlapping patch results."""
    merged = []
    for class_name in CLASS_NAMES:
        candidates = sorted(
            [p for p in predictions if p["predicted_class"] == class_name],
            key=lambda item: item["confidence"],
            reverse=True,
        )
        kept = []
        for candidate in candidates:
            if all(
                _box_iou(candidate["bbox"], previous["bbox"]) < iou_threshold
                for previous in kept
            ):
                kept.append(candidate)
        merged.extend(kept)

    return sorted(merged, key=lambda item: item["confidence"], reverse=True)


def _make_gradcam(model, patch_image, class_index):
    input_tensor = transform(patch_image).unsqueeze(0).to(DEVICE)
    display_image = patch_image.convert("RGB").resize((224, 224))
    rgb_float = np.asarray(display_image, dtype=np.float32) / 255.0

    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    try:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(class_index)],
        )[0]
        overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    finally:
        del cam

    return _to_data_url(overlay)


def model_status():
    return {
        "model": "ResNet-34",
        "dataset": "NEU Surface Defect Database",
        "classes": CLASS_NAMES,
        "device": str(DEVICE),
        "checkpoint_present": MODEL_PATH.exists(),
        "full_sheet_inspection": True,
        "patch_size": PATCH_SIZE,
        "patch_stride": PATCH_STRIDE,
    }


def analyze_image(image_bytes: bytes, filename: str):
    """Existing single-image endpoint. Kept compatible with the current frontend."""
    model = load_model()
    started = time.perf_counter()

    original = Image.open(io.BytesIO(image_bytes)).convert("L")
    display_image = original.convert("RGB").resize((224, 224))
    rgb_float = np.asarray(display_image, dtype=np.float32) / 255.0
    input_tensor = transform(original).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = int(probabilities.argmax().item())
    probability_dict = {
        CLASS_NAMES[i]: round(float(probabilities[i].item()), 6)
        for i in range(len(CLASS_NAMES))
    }

    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    try:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(predicted_index)],
        )[0]
        overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    finally:
        del cam

    elapsed_ms = (time.perf_counter() - started) * 1000

    return {
        "filename": filename,
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": round(float(probabilities[predicted_index].item()), 6),
        "probabilities": probability_dict,
        "original_image": _to_data_url(np.asarray(display_image)),
        "gradcam_overlay": _to_data_url(overlay),
        "inference_time_ms": round(elapsed_ms, 2),
        "note": "Grad-CAM is an approximate explanation, not a precise defect boundary.",
    }


def analyze_full_sheet(
    image_bytes: bytes,
    filename: str,
    confidence_threshold: float = 0.60,
):
    """
    Classify overlapping patches across a full sheet.
    The threshold is a candidate filter, not a calibrated defect/no-defect test.
    """
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1.")

    model = load_model()
    started = time.perf_counter()

    original = Image.open(io.BytesIO(image_bytes)).convert("L")
    width, height = original.size
    patches = _extract_patches(original)

    all_patch_predictions = []
    for batch_start in range(0, len(patches), INFERENCE_BATCH_SIZE):
        batch = patches[batch_start:batch_start + INFERENCE_BATCH_SIZE]
        tensors = torch.stack([transform(item["image"]) for item in batch]).to(DEVICE)

        with torch.inference_mode():
            logits = model(tensors)
            batch_probabilities = torch.softmax(logits, dim=1).cpu().numpy()

        for item, probabilities in zip(batch, batch_probabilities):
            predicted_index = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_index])
            all_patch_predictions.append({
                "predicted_class": CLASS_NAMES[predicted_index],
                "confidence": confidence,
                "class_index": predicted_index,
                "probabilities": {
                    CLASS_NAMES[i]: round(float(probabilities[i]), 6)
                    for i in range(len(CLASS_NAMES))
                },
                "bbox": item["bbox"],
                "patch_image": item["image"],
            })

    candidates = [
        item for item in all_patch_predictions
        if item["confidence"] >= confidence_threshold
    ]
    merged = _merge_overlapping_predictions(candidates)

    # Draw approximate patch boxes on a resized copy of the original sheet.
    max_preview_side = 1600
    scale = min(1.0, max_preview_side / max(width, height))
    preview_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    preview = original.convert("RGB").resize(preview_size)
    draw = ImageDraw.Draw(preview)

    class_colors = {
        "crazing": (255, 80, 80),
        "inclusion": (255, 170, 0),
        "patches": (80, 170, 255),
        "pitted_surface": (190, 100, 255),
        "rolled-in_scale": (255, 220, 60),
        "scratches": (60, 220, 170),
    }

    for detection in merged:
        x1, y1, x2, y2 = detection["bbox"]
        scaled_box = (
            int(x1 * scale), int(y1 * scale),
            int(x2 * scale), int(y2 * scale),
        )
        color = class_colors[detection["predicted_class"]]
        draw.rectangle(scaled_box, outline=color, width=3)
        label = "{} {:.0f}%".format(
            detection["predicted_class"],
            detection["confidence"] * 100,
        )
        text_y = max(0, scaled_box[1] - 16)
        draw.text((scaled_box[0], text_y), label, fill=color)

    # One representative Grad-CAM per class, limited to keep response size manageable.
    explanations = []
    best_by_class = {}
    for detection in merged:
        class_name = detection["predicted_class"]
        if class_name not in best_by_class:
            best_by_class[class_name] = detection

    for class_name, detection in best_by_class.items():
        if len(explanations) >= MAX_GRADCAM_EXPLANATIONS:
            break
        patch = next(
            item["patch_image"] for item in all_patch_predictions
            if item["bbox"] == detection["bbox"]
            and item["predicted_class"] == class_name
        )
        cam_url = _make_gradcam(model, patch, detection["class_index"])
        explanations.append({
            "predicted_class": class_name,
            "confidence": round(detection["confidence"], 6),
            "bbox": detection["bbox"],
            "gradcam_overlay": cam_url,
        })

    class_summary = {}
    for class_name in CLASS_NAMES:
        class_detections = [
            item for item in merged if item["predicted_class"] == class_name
        ]
        class_summary[class_name] = {
            "candidate_regions": len(class_detections),
            "highest_confidence": (
                round(max(item["confidence"] for item in class_detections), 6)
                if class_detections else None
            ),
        }

    public_detections = [
        {
            "predicted_class": item["predicted_class"],
            "confidence": round(item["confidence"], 6),
            "bbox": item["bbox"],
            "probabilities": item["probabilities"],
        }
        for item in merged
    ]

    elapsed_ms = (time.perf_counter() - started) * 1000

    return {
        "filename": filename,
        "image_width": width,
        "image_height": height,
        "patch_size": PATCH_SIZE,
        "patch_stride": PATCH_STRIDE,
        "total_patches": len(patches),
        "candidate_patches_before_merging": len(candidates),
        "candidate_regions_after_merging": len(merged),
        "confidence_threshold": confidence_threshold,
        "class_summary": class_summary,
        "detections": public_detections,
        "sheet_visualization": _to_data_url(np.asarray(preview)),
        "gradcam_explanations": explanations,
        "inference_time_ms": round(elapsed_ms, 2),
        "important_limitation": (
            "The model has no defect-free class and was trained on close-up NEU "
            "defect images. Confidence filtering only selects candidate patches; "
            "it does not establish that a region is defective or defect-free. "
            "Grad-CAM and patch boxes are approximate, not precise boundaries."
        ),
    }
