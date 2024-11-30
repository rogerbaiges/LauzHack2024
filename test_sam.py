import os
import sys
current_dir = os.getcwd()

# Add the parent directory of 'samsam' to the system path
sys.path.append(os.path.abspath(os.path.join(current_dir, 'samsam')))
from samsam.sam2.automatic_mask_generator import SamAutomaticMaskGenerator
from samsam.sam2.utils import sam_model_registry
import numpy as np
import torch
import cv2
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2


model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
IMAGE_PATH = "olives.png"
TEXT_PROMPT = "single olive."
BOX_TRESHOLD = 0.2
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD,
    device="cpu"
)

print(boxes)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("annotated_olives.jpg", annotated_frame)

# Load SAM2 model
sam_checkpoint = "weights/sam_vit_h_4b8939.pth"  # Path to SAM2 model weights
sam_model_type = "vit_h"  # Model type
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# Initialize SAM's automatic mask generator
mask_generator = SamAutomaticMaskGenerator(sam)

# Convert GroundingDINO boxes to SAM2 format
# GroundingDINO boxes are in (x_min, y_min, x_max, y_max) format.
# SAM expects boxes in (x_center, y_center, width, height) format, normalized to [0, 1].
height, width, _ = image_source.shape
boxes_sam = []
for box in boxes:
    x_min, y_min, x_max, y_max = box
    x_center = (x_min + x_max) / 2 / width
    y_center = (y_min + y_max) / 2 / height
    box_width = (x_max - x_min) / width
    box_height = (y_max - y_min) / height
    boxes_sam.append([x_center, y_center, box_width, box_height])

# Convert normalized boxes to pixel space for visualization and further processing
pixel_boxes = []
for box in boxes:
    x_min, y_min, x_max, y_max = map(int, box)
    pixel_boxes.append([x_min, y_min, x_max, y_max])

# Generate masks using SAM
masks = []
for pixel_box in pixel_boxes:
    x_min, y_min, x_max, y_max = pixel_box
    region = image_source[y_min:y_max, x_min:x_max]
    region_masks = mask_generator.generate(region)
    masks.append(region_masks)

# Save the masks and optionally annotate the image
for i, region_masks in enumerate(masks):
    for j, mask in enumerate(region_masks):
        mask_image = mask["segmentation"] * 255  # Convert mask to binary image
        mask_path = f"mask_{i}_{j}.png"
        cv2.imwrite(mask_path, mask_image)

# Annotate the original image with segmentation results
for pixel_box in pixel_boxes:
    x_min, y_min, x_max, y_max = pixel_box
    cv2.rectangle(image_source, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

cv2.imwrite("segmented_olives.jpg", image_source)
