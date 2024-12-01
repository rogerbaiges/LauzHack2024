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
from skimage.filters import threshold_otsu, gaussian
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.feature import canny
from skimage.morphology import closing, square
from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.color import rgb2hsv
from scipy import ndimage
from skimage import measure
from sklearn.cluster import KMeans
class ImageSegmenter:
    def __init__(self, grounding_dino_cfg, grounding_dino_weights, sam_cfg, sam_weights, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        
        self.grounding_dino_model = load_model(grounding_dino_cfg, grounding_dino_weights)
        
        self.sam_model = build_sam2(sam_cfg, sam_weights, device=self.device)
        self.sam_predictor = SAM2ImagePredictor(self.sam_model)
        self.people_counting_model = self._load_people_counting_model()
    def _load_people_counting_model(self):
        from ultralytics import YOLO
        
        return YOLO("yolo11n.pt")  
    
    def extract_edges(self, image_path=None, sigma=1.5):
        if image_path:
            self.image_path = image_path
            self.image = np.array(Image.open(self.image_path).convert("RGB"))
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        edges = canny(gray_image, sigma=sigma)
        return edges
    def extract_corners(self, image_path=None):
        if image_path:
            self.image_path = image_path
            self.image = np.array(Image.open(self.image_path).convert("RGB"))
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        corners = cv2.goodFeaturesToTrack(gray_image, maxCorners=100, qualityLevel=0.3, minDistance=7)
        if corners is not None:
            corners = np.int0(corners)
            return corners
        else:
            return []
    def extract_sift_features(self, image_path=None):
        if image_path:
            self.image_path = image_path
            self.image = np.array(Image.open(self.image_path).convert("RGB"))
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        return keypoints, descriptors
    def extract_orb_features(self, image_path=None):
        if image_path:
            self.image_path = image_path
            self.image = np.array(Image.open(self.image_path).convert("RGB"))
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray_image, None)
        return keypoints, descriptors
    def match_features(self, descriptors1, descriptors2, method='bf', cross_check=True, ratio_threshold=0.75):
        if descriptors1 is None or descriptors2 is None:
            print("Warning: One or both descriptor sets are None. Skipping matching.")
            return []
        if descriptors1.shape[1] != descriptors2.shape[1]:
            print(f"Error: Descriptor matrices have different numbers of columns ({descriptors1.shape[1]} vs {descriptors2.shape[1]}). Cannot match.")
            return []
        if method == 'bf':
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=cross_check)
            
            descriptors1 = descriptors1.astype(np.float32)
            descriptors2 = descriptors2.astype(np.float32)
            try:
                matches = bf.match(descriptors1, descriptors2)
                matches = sorted(matches, key=lambda x: x.distance)
            except cv2.error as e:
                print(f"Error during BFMatcher: {e}")
                return []
        elif method == 'flann':
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            try:
                matches = flann.knnMatch(descriptors1, descriptors2, k=2)
                good = []
                for m, n in matches:
                    if m.distance < ratio_threshold * n.distance:
                        good.append(m)
                matches = good
            except cv2.error as e:
                print(f"Error during FLANN matching: {e}")
                return []
        else:
            raise ValueError("Invalid matching method. Choose 'bf' or 'flann'.")
        return matches
    
    def draw_matches(self, img1_path, keypoints1, img2_path, keypoints2, matches):
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img_matches)
        plt.show()
        
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
        image_path = self.image_path 
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
        box = self.xyxy_boxes[0]  
        x_min, y_min, x_max, y_max = box
        input_box = np.array([x_min, y_min, x_max, y_max])
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=True,  
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
    
    def segment_watershed(self, image_path=None, gaussian_blur=3, closing_size=3):
        if image_path:
            self.image_path = image_path
            try:
                self.image = np.array(Image.open(self.image_path).convert("RGB"))
            except FileNotFoundError:
                print(f"Error: Image file not found at {image_path}")
                return None
        else:
            try:
                self.image = np.array(Image.fromarray(self.image).convert("RGB"))
            except AttributeError:
                print(f"Error: No image loaded. Use load_and_predict_dino first or provide image path.")
                return None
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        
        blurred_image = gaussian(gray_image, sigma=gaussian_blur)
        
        thresh = threshold_otsu(blurred_image)
        
        binary = blurred_image > thresh
        
        binary = closing(binary, square(closing_size))
        
        print("Shape of blurred_image:", blurred_image.shape)
        print("Shape of binary mask:", binary.shape)
        
        distance = ndimage.distance_transform_edt(binary)
        
        local_maxi = peak_local_max(distance, footprint=np.ones((3, 3)), labels=binary)
        
        if local_maxi.size == 0:
            print("No local maxima found in distance transform.")
            return None
        
        markers = np.zeros_like(distance, dtype=int)
        markers[tuple(local_maxi.T)] = np.arange(1, len(local_maxi) + 1)
        
        print("Shape of distance transform:", distance.shape)
        print("Shape of markers:", markers.shape)
        print("Shape of binary mask (again):", binary.shape)
        try:
            labels = watershed(-distance, markers, mask=binary)
            self.watershed_segments = labels
            return labels
        except ValueError as e:
            print(f"Error in watershed segmentation: {e}")
            print("Shape mismatch likely between markers and binary mask.")
            self.watershed_segments = None
            return None
    
    
    def count_people(self, image_path=None):
        if image_path:
            self.image_path = image_path
        
        image = cv2.imread(self.image_path)
        people_count, density = LWCC.get_count(self.image_path, return_density=True)
        plt.imshow(density)
        plt.show()
        return people_count
    def detect_objects_yolo(self, image_path=None, model_path="yolov8n.pt", conf_threshold=0.5):
      from ultralytics import YOLO
      if image_path:
          self.image_path = image_path
          self.image = np.array(Image.open(self.image_path).convert("RGB"))
      else:
          self.image = np.array(Image.fromarray(self.image).convert("RGB"))
      model = YOLO(model_path)
      results = model(self.image)
      detections = []
      for r in results:
          boxes = r.boxes  
          for box in boxes:
              b = box.xyxy[0]  
              c = int(box.cls)  
              conf = box.conf[0]  
              if conf > conf_threshold:
                  detections.append({
                      "box": b.tolist(),
                      "class_id": c,
                      "class_name": model.names[c],
                      "confidence": conf.item(),
                  })
      return detections
    
    def analyze_crop_coverage(self, image_path, segmentation_method='felzenszwalb', num_colors=5, visualize=True):
        """
        Analyzes an aerial image to estimate crop coverage in fields.
        Args:
            image_path: Path to the aerial image.
            segmentation_method: 'felzenszwalb', 'slic', or 'quickshift'.
            visualize: Whether to display intermediate and final results.
        Returns:
            Estimated crop coverage percentage (0-100).
            Also can return an annotated image for visualization (optional).
        """
        
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_float = img_as_float(img)  
        
        if segmentation_method == 'felzenszwalb':
            segments = felzenszwalb(img_float, scale=100, sigma=0.5, min_size=50)
        elif segmentation_method == 'slic':
            segments = slic(img_float, n_segments=250, compactness=30, sigma=1,
                            start_label=1)
        elif segmentation_method == 'quickshift':
            segments = quickshift(img_float, ratio=0.5, kernel_size=3, max_dist=6)
        else:
            raise ValueError("Invalid segmentation method.")
        if visualize:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(mark_boundaries(img_float, segments))
            plt.show()
        segment_colors = self.create_palette(image_path, num_colors)
        if visualize:
            
            color_palette = np.zeros((50, num_colors * 50, 3), dtype=np.uint8)
            for i, color in enumerate(segment_colors):
                color_palette[:, i * 50:(i + 1) * 50, :] = color
            plt.imshow(color_palette)
            plt.title("Select the crop color index (0-{}):".format(num_colors -1))
            plt.show()
            crop_color_index = int(input("Enter the crop color index: "))
            crop_color = segment_colors[crop_color_index]
            
        field_segments = []
        for segment_label in np.unique(segments):
            segment_mask = (segments == segment_label)
            
            contours, _ = cv2.findContours(segment_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                x, y, w, h = cv2.boundingRect(contours[0])
                aspect_ratio = float(w) / h
                
                if 0.5 < aspect_ratio < 2:
                    field_segments.append({"label": segment_label, "rect": (x, y, w, h)})
        
        hsv_img = rgb2hsv(img)
        total_field_area = 0
        cropped_field_area = 0
        high_saturation_mask = np.zeros_like(segments, dtype=bool)
        
        total_field_area = 0
        cropped_field_area = 0
        for field in field_segments:
            mask = segments == field['label']
            segment_color_hsv = rgb2hsv(np.uint8([[crop_color]]))[0, 0] 
            average_segment_color_hsv = np.mean(rgb2hsv(img[mask]), axis=0) 
            print(average_segment_color_hsv)
            print(segment_color_hsv)
            print("---")
            total_field_area += np.sum(mask)
            if np.linalg.norm(average_segment_color_hsv - segment_color_hsv) < 0.3: 
                
                cropped_field_area += np.sum(mask)
                
                
                high_saturation_mask |= mask
        
        if visualize:
            
            visualization_img = img.copy()
            
            
            visualization_img[high_saturation_mask] = [0, 255, 0]  
            
            
            plt.imshow(visualization_img)
            plt.axis('off')  
            plt.show()
        
        crop_coverage = (cropped_field_area / total_field_area) * 100 if total_field_area else 0
        
        if visualize and total_field_area > 0:
            print(crop_coverage)
        return crop_coverage
    def draw_detections(self, detections, image_path=None, draw_labels=True):
        if image_path:
            self.image_path = image_path
            self.image = cv2.imread(self.image_path)
        else:
            self.image = cv2.cvtColor(np.array(Image.fromarray(self.image).convert("RGB")), cv2.COLOR_RGB2BGR)
        for detection in detections:
            box = detection["box"]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            class_name = detection["class_name"]
            confidence = detection["confidence"]
            
            color = (0, 255, 0)
            thickness = 2
            cv2.rectangle(self.image, (x1, y1), (x2, y2), color, thickness)
            
            if draw_labels:
                label = f"{class_name}: {confidence:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                text_size = cv2.getTextSize(label, font, font_scale, thickness=1)[0]
                text_x = x1
                text_y = y1 - 5 if y1 - 5 > text_size[1] else y1 + 5 + text_size[1]
                cv2.rectangle(self.image, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), color, cv2.FILLED)
                cv2.putText(self.image, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness=1)
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    def create_palette(self, image_path, num_colors=5, display=False):
        """
        Generate a color palette with predominant colors from an image.
        
        Parameters:
        image_path (str): Path to the input image.
        num_colors (int): Number of colors in the palette (default: 5).
        
        Returns:
        List[Tuple[int, int, int]]: List of RGB tuples representing the color palette.
        """
        
        image = Image.open(image_path)
        image = image.convert('RGB')  
        image = image.resize((150, 150))  
        
        
        image_data = np.array(image)
        pixels = image_data.reshape(-1, 3)  
        
        
        kmeans = KMeans(n_clusters=num_colors, random_state=0)
        kmeans.fit(pixels)
        
        
        palette = [tuple(map(int, color)) for color in kmeans.cluster_centers_]
        
        
        if display:
            self._display_palette(palette)
        
        return palette
    def _display_palette(self, palette):
        """
        Display the generated color palette as a horizontal bar.
        
        Parameters:
        palette (List[Tuple[int, int, int]]): List of RGB colors.
        """
        palette_image = np.zeros((50, 50 * len(palette), 3), dtype=np.uint8)
        for i, color in enumerate(palette):
            palette_image[:, i * 50:(i + 1) * 50] = color
        plt.figure(figsize=(8, 2))
        plt.axis('off')
        plt.imshow(palette_image)
        plt.show()
    
    def count_people(self, image_path=None):
        if image_path:
            self.image_path = image_path
        
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
    def visualize_segmentation(self, colormap='viridis'):
        plt.imshow(self.watershed_segments, cmap=colormap)
        plt.axis('off')
        plt.show()
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
                    break  
        return found_images
if __name__ == "__main__":
    
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
    """
    """
    edges = segmenter.extract_edges(image_path="reference.png")
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    plt.show()
    corners = segmenter.extract_corners(image_path="reference.png")
    if corners is not None:
        img_with_corners = cv2.imread("reference.png")
        for i in corners:
            x, y = i.ravel()
            cv2.circle(img_with_corners, (x, y), 3, 255, -1)
        plt.imshow(cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    else:
        print("No corners detected")
    keypoints_sift, descriptors_sift = segmenter.extract_sift_features("reference.png")
    keypoints_orb, descriptors_orb = segmenter.extract_orb_features("reference_rotated.png")
    img_sift = cv2.imread("reference.png")
    img_sift_keypoints = cv2.drawKeypoints(img_sift, keypoints_sift, img_sift)
    plt.imshow(cv2.cvtColor(img_sift_keypoints, cv2.COLOR_BGR2RGB))
    plt.show()
    img_orb = cv2.imread("reference_rotated.png")
    img_orb_keypoints = cv2.drawKeypoints(img_orb, keypoints_orb, img_orb)
    plt.imshow(cv2.cvtColor(img_orb_keypoints, cv2.COLOR_BGR2RGB))
    plt.show()
    matches_sift = segmenter.match_features(descriptors_sift, descriptors_orb, method='bf')
    segmenter.draw_matches("reference.png", keypoints_sift,"reference_rotated.png", keypoints_orb, matches_sift[:20])
    segmenter.segment_watershed("coins.png")
    segmenter.visualize_segmentation()
    detections = segmenter.detect_objects_yolo(image_path="faces/cai3.jpg", model_path="yolo11n.pt")
    segmenter.draw_detections(detections)
    segmenter.detect_objects_yolo(model_path="yolo11n.pt")
    detections_2 = segmenter.detect_objects_yolo(conf_threshold = 0.3)
    segmenter.draw_detections(detections_2)
    """
    image_path = "crops2.png"  
    crop_coverage = segmenter.analyze_crop_coverage(image_path)
    print(f"Estimated crop coverage: {crop_coverage:.2f}%")
    
    palette = segmenter.create_palette("faces/cai3.jpg", num_colors=5, display=True)