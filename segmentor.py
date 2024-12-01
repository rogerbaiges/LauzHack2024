import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.ops import box_convert
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import os
import sys
from lwcc import LWCC
current_dir = os.getcwd()
import face_recognition


class ImageSegmenter:
    def __init__(self, grounding_dino_cfg, grounding_dino_weights, sam_cfg, sam_weights, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"

        # Load GroundingDINO model
        self.grounding_dino_model = load_model(grounding_dino_cfg, grounding_dino_weights)

        # Load SAM2 model
        self.sam_model = build_sam2(sam_cfg, sam_weights, device=self.device)
        self.sam_predictor = SAM2ImagePredictor(self.sam_model)
        self.people_counting_model = self._load_people_counting_model()

    def _load_people_counting_model(self):
        from ultralytics import YOLO
        # Load a pre-trained YOLOv8 model fine-tuned for person detection
        return YOLO("yolo11n.pt")  # Replace with a specific model if needed


    def load_and_predict_dino(self, image_path, text_prompt, box_threshold=0.2, text_threshold=0.25):
        self.image_path = image_path
        self.image_source, self.image = load_image(image_path)
        self.text_prompt = text_prompt

        self.boxes, self.logits, self.phrases = predict(
            model=self.grounding_dino_model,
            image=self.image,
            caption=self.text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
            remove_combined=True
        )

        image_path = self.image_path # Replace with your image
        image = Image.open(image_path).convert("RGB")
        self.image = np.array(image)

        self.height, self.width = self.image.shape[:2]
        self.xyxy_boxes = box_convert(
            boxes=self.boxes * torch.Tensor([self.width, self.height, self.width, self.height]),
            in_fmt="cxcywh",
            out_fmt="xyxy"
        ).numpy()

        print(self.xyxy_boxes)

        return self.xyxy_boxes

    def segment_masks(self, area_threshold=100, iou_threshold=0.5):
        image = Image.open(self.image_path).convert("RGB")
        image = np.array(image)
        self.image = image
        self.sam_predictor.set_image(image)
        complete_masks = []
        for box in self.xyxy_boxes:
            x_min, y_min, x_max, y_max = box
            input_box = np.array([x_min, y_min, x_max, y_max])
            masks, scores, _ = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            complete_masks.extend(masks)

        self.filtered_masks = self._remove_small_masks(complete_masks, area_threshold)
        self.unique_masks = self._eliminate_duplicates(self.filtered_masks, iou_threshold)
        return self.unique_masks

    def _remove_small_masks(self, masks, area_threshold):
        filtered_masks = []
        for mask in masks:
            # Calculate the area of the mask (sum of the mask)
            mask_area = np.sum(mask)
            if mask_area >= area_threshold:
                
                filtered_masks.append(mask)
            else:
                print("too small")
        return filtered_masks

    def _eliminate_duplicates(self, masks, iou_threshold):
        unique_masks = []
        for mask in masks:
            if not any(self._compute_iou(mask, um) > iou_threshold for um in unique_masks):
                unique_masks.append(mask)
        return unique_masks

    def _compute_iou(self, mask1, mask2):
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        return np.sum(intersection) / np.sum(union)

    def count_masks(self):
        return len(self.unique_masks)

    def segment_first_mask_variations(self):
        image = Image.open(self.image_path).convert("RGB")
        image = np.array(image)
        self.image = image
        self.sam_predictor.set_image(image)
        box = self.xyxy_boxes[0]  # Use the first bounding box
        x_min, y_min, x_max, y_max = box
        input_box = np.array([x_min, y_min, x_max, y_max])

        masks, scores, _ = self.sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=True,  # Generate multiple masks
        )

        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10, 10))
            plt.imshow(np.array(self.image))
            self._show_mask(mask, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()

    def apply_masks_to_image(self, color=(255, 0, 0)):
        modified_image = np.array(self.image)
        for mask in self.unique_masks:
            modified_image = self._apply_realistic_color_to_mask(modified_image, mask, color)
        return modified_image

    def _apply_realistic_color_to_mask(self, image, mask, color, blur_radius=15):
        mask = mask.astype(bool)
        image_with_color = image.copy()
        color_overlay = np.zeros_like(image)
        color_overlay[mask] = color
        color_overlay_blurred = cv2.GaussianBlur(color_overlay, (blur_radius, blur_radius), 0)
        luminance = np.mean(image, axis=2) / 255
        luminance = np.clip(luminance, 0.1, 1.0)

        for i in range(3):
            image_with_color[:, :, i] = image_with_color[:, :, i] * (1 - mask * luminance) + \
                                        color_overlay_blurred[:, :, i] * mask * luminance
        return image_with_color
    
    def count_people(self, image_path=None):
        if image_path:
            self.image_path = image_path

        # Read the image
        image = cv2.imread(self.image_path)

        people_count, density = LWCC.get_count(self.image_path, return_density=True)

        plt.imshow(density)
        plt.show()

        return people_count

    def _show_mask(self, mask, ax, random_color=False):
        color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def find_person_in_images(self, reference_image_path, images_paths):
        """
        Finds images containing the same person as in the reference image.

        Args:
            reference_image_path (str): Path to the image of the person to find.
            images_paths (list of str): Paths to the images to search in.

        Returns:
            list of str: Paths of images containing the person.
        """

        reference_image = face_recognition.load_image_file(reference_image_path)
        try:
            reference_face_encoding = face_recognition.face_encodings(reference_image)[0]
        except IndexError:
            print("No face found in reference image.")
            return []

        known_faces = [reference_face_encoding]
        found_images = []

        for image_path in images_paths:
            img = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(img)
            face_encodings = face_recognition.face_encodings(img, face_locations)

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_faces, face_encoding)
                if True in matches:
                    found_images.append(image_path)
                    break  # If face found, no need to check other faces in the image

        return found_images

if __name__ == "__main__":
    # Add the parent directory of 'samsam' to the system path
    sys.path.append(os.path.abspath(os.path.join(current_dir, './samsam/')))

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    import numpy as np
    import torch
    import cv2
    from PIL import Image
    import matplotlib.pyplot as plt

    from hydra import initialize_config_module
    from hydra.core.global_hydra import GlobalHydra


    if not GlobalHydra.instance().is_initialized():
        initialize_config_module("samsam/sam2", version_base="1.2")

    segmenter = ImageSegmenter(
        grounding_dino_cfg="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        grounding_dino_weights="GroundingDINO/weights/groundingdino_swint_ogc.pth",
        sam_cfg="C:/Users/Usuario/Documents/Projectes/LauzHack2024/LauzHack2024/samsam/sam2/configs/sam2.1/sam2.1_hiera_t.yaml",
        sam_weights="C:/Users/Usuario/Documents/Projectes/LauzHack2024/LauzHack2024/samsam/weights/sam2.1_hiera_tiny.pt"
    )

    """
    image_path = "crowd2.png"
    people_count = segmenter.count_people(image_path=image_path)
    print(f"Number of people detected: {people_count}")
    """

    """
    boxes = segmenter.load_and_predict_dino("deposits.png", "circle.")
    masks = segmenter.segment_masks()

    print(f"Number of masks: {segmenter.count_masks()}")
    segmenter.segment_first_mask_variations()

    final_image = segmenter.apply_masks_to_image(color=(0, 255, 0))
    plt.imshow(final_image)
    plt.axis('off')
    plt.show()
    """


    reference_image_path = "reference.png"
    faces_folder = "faces"
    faces_images_paths = [os.path.join(faces_folder, img) for img in os.listdir(faces_folder) if img.endswith((".png", ".jpg", ".jpeg"))]

    matching_images = segmenter.find_person_in_images(reference_image_path, faces_images_paths)

    if matching_images:
        print("Images containing the same person as in 'reference.png':")
        for image_path in matching_images:
            print(image_path)
            plt.imshow(Image.open(image_path))
            plt.show()
    else:
        print("No matching images found.")
    
