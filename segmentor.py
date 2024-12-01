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
	


	def get_user_click(self, image):
		"""Gets the coordinates of a user's click on an image."""

		fig, ax = plt.subplots()
		ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # Convert to RGB for matplotlib

		click_coordinates = []

		def onclick(event):
			click_coordinates.append((int(event.xdata), int(event.ydata)))
			plt.close() # Close the plot after the click

		fig.canvas.mpl_connect('button_press_event', onclick)
		plt.show()

		return click_coordinates[0] if click_coordinates else None
	def extract_mask_with_sam2(self, image, click_point):
		"""Extracts a mask using SAM2 based on the user's click."""

		sam2 = self.sam_model
		predictor = self.sam_predictor
		predictor.set_image(image) 
		masks, _, _ = predictor.predict(
			point_coords=np.array([click_point]),  # Click coordinates as a NumPy array
			point_labels=np.array([1]),              # Positive label (1 for foreground)
			multimask_output=False,
		)


		plt.imshow(masks[0])
		plt.title("Extracted Mask")
		plt.show()

		return masks[0]
	
	def find_similar_features_in_image(self, image, mask, size_tolerance=0.8, rotation_tolerance=90):
		"""Finds similar features in the SAME image based on the provided mask."""

		mask = mask.astype(np.uint8) * 255  # Convert to 8-bit

		# Get the bounding box of the template feature from the mask
		contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		tx, ty, tw, th = cv2.boundingRect(contours[0])
		template = image[ty:ty+th, tx:tx+tw] # Extract template from image

		matched_regions = []

		for scale in np.linspace(0.9, 1.10, 10):  # Scale to handle size variations
			resized_template = cv2.resize(template, None, fx=scale, fy=scale)
			
			for angle in range(0, rotation_tolerance + 1, 30): # Steps of 10 degrees
				rows, cols = resized_template.shape[:2]
				rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
				rotated_template = cv2.warpAffine(resized_template, rotation_matrix, (cols, rows))


				result = cv2.matchTemplate(image, rotated_template, cv2.TM_CCOEFF_NORMED)
				threshold = 0.85 # Adjust as needed
				loc = np.where( result >= threshold)

				for pt in zip(*loc[::-1]):
					w, h = rotated_template.shape[1], rotated_template.shape[0]  # Use rotated template size.
					if abs(w - tw) <= size_tolerance and abs(h - th) <= size_tolerance  and (pt[0] != tx or pt[1] != ty): 
						matched_regions.append((pt[0], pt[1], w, h, angle)) # Store angle too

		return matched_regions
	
	def find_from_click(self, image_path):
		image = cv2.imread(image_path)
		click_point = self.get_user_click(image)
		if click_point is None:
			print("No click received. Exiting.")
			sys.exit()

		mask = self.extract_mask_with_sam2(image, click_point)
		tolerance = 5 # Adjust as needed.
		similar_regions = self.find_similar_features_in_image(image, mask, tolerance)

		if similar_regions:
			print("Similar features found:")
			for x, y, w, h, _ in similar_regions:
				cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
			cv2.imshow("Detected Features", image)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
		else:
			print("No similar features found.")

	def remove_from_click(self, image_path):
		image = cv2.imread(image_path)
		click_point = self.get_user_click(image)
		if click_point is None:
			print("No click received. Exiting.")
			sys.exit()

		mask = self.extract_mask_with_sam2(image, click_point)

		inpainted_image = self.remove_and_inpaint_cv2(image.copy(), mask)



		# --- Visualization ---
		plt.figure(figsize=(10, 5))
		plt.subplot(1, 2, 1)
		plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) #Original image
		plt.title("Original Image with Mask")
		plt.subplot(1, 2, 2)
		plt.imshow(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB))
		plt.title("Inpainted Image")
		plt.show()

	def get_segmented_image(self):
		"""Creates an image containing only the segmented objects (masks)."""

		segmented_image = np.zeros_like(self.image)

		for mask in self.unique_masks:
			bool_mask = mask.astype(bool)
			
			segmented_image[bool_mask] = self.image[bool_mask]  # Copy masked parts to the new image

		return segmented_image
	
	def count_masks_by_color(self, target_color, tolerance=60, enhance_contrast = True):
		"""Counts masks containing elements with a similar color to the target color.

		Args:
			target_color (tuple): RGB target color tuple (e.g., (255, 0, 0) for red).
			tolerance (int): Tolerance for color similarity (default: 30).

		Returns:
			int: Number of masks matching the color criteria.
		"""

		matching_masks_count = 0
		enhanced_image = self._enhance_contrast(self.image)
		for mask in self.unique_masks:
			bool_mask = mask.astype(bool)
			masked_image = enhanced_image[bool_mask]  # Apply mask to image
		   
													  
			pixels = masked_image.reshape(-1, 3)
			kmeans = KMeans(n_clusters=1, random_state=0)  # Find the dominant color
			kmeans.fit(pixels)
			dominant_color = tuple(map(int, kmeans.cluster_centers_[0]))
			print(dominant_color)

			color_distance = np.linalg.norm(np.array(dominant_color) - np.array(target_color))

			if color_distance <= tolerance:
				matching_masks_count += 1

		return matching_masks_count
	def _enhance_contrast(self, image):
		"""Enhances contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)."""

		lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
		l, a, b = cv2.split(lab)

		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
		l_enhanced = clahe.apply(l)

		lab_enhanced = cv2.merge((l_enhanced, a, b))
		image_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

		return image_enhanced


	def remove_and_inpaint_cv2(self, image, mask):
		"""Removes the masked region and inpaints it using OpenCV's inpainting methods."""

		mask = mask.astype(np.uint8) * 255  # Ensure mask is 8-bit
		image = image.astype(np.uint8)  # Image should also be 8-bit

		# OpenCV inpainting (choose one of the methods below)
		inpainted_image = cv2.inpaint(image, mask, inpaintRadius=0.5, flags=cv2.INPAINT_TELEA) # Telea
		# or
		# inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)  # Navier-Stokes


		return inpainted_image
	
	def transcribe_text_from_image(self, image_path=None, model_path="path/to/ocr/model"):
		"""
		Transcribes text from an image using OCR.

		Args:
			image_path (str, optional): Path to the image. If None, uses the currently loaded image. Defaults to None.
			model_path (str, optional): Path to the OCR model. Defaults to "path/to/ocr/model". You'll need to replace this with the actual path. 

		Returns:
			str: The transcribed text.
		"""
		try:
			import easyocr  # You'll need to install easyocr: pip install easyocr
		except ImportError:
			print("easyocr is not installed. Please install it using: pip install easyocr")
			return ""

		if image_path:
			self.image_path = image_path
			image = cv2.imread(self.image_path)
		elif hasattr(self, 'image'):  # Check if an image is already loaded
			image = self.image
		else:
			print("No image loaded or provided. Please load an image first.")
			return ""

		reader = easyocr.Reader(['en']) # Initialize reader with preferred languages
		result = reader.readtext(image)


		transcribed_text = ""
		for (bbox, text, prob) in result:
			 transcribed_text += text + " "
		
		return transcribed_text.strip()
	
	def analyze_fuel_level(self, image_path=None, visualize=True):
		"""
		Analyzes images of fuel gauges to estimate the fuel level. This could be used 
		for remote monitoring of fuel tanks at petrol stations or detecting anomalies.

		Args:
			image_path (str, optional): Path to the image. If None, uses the current image. Defaults to None.
			visualize (bool, optional): Whether to display intermediate steps. Defaults to False.

		Returns:
			float: Estimated fuel level (0.0 to 1.0).  Returns -1 if gauge detection fails.
		"""
		if image_path:
			self.image_path = image_path
			image = cv2.imread(self.image_path)
		elif hasattr(self, 'image'):
			image = self.image
		else:
			print("No image loaded or provided.")
			return -1  # Or raise an exception

		gauge_bbox = self.detect_gauge(image)  # Replace with your gauge detection logic

		if gauge_bbox is None:
			print("Fuel gauge not detected. Using the whole image.")
			x, y, w, h = 0, 0, image.shape[1], image.shape[0]  # Use whole image dimensions
		else:
			x, y, w, h = gauge_bbox

			x = max(x, 0)
			y = max(y, 0)
			x = min(x, image.shape[1])
			y = 0 if y > image.shape[0] else y

			w = min(w, image.shape[1])
			h = min(h, image.shape[0])


		print(gauge_bbox)
		gauge_image = image[y:y + h, x:x + w]

		# 2. Convert to HSV and threshold for needle color (e.g., red).
		hsv = cv2.cvtColor(gauge_image, cv2.COLOR_BGR2HSV)
		lower_red = np.array([0, 50, 50])    # Adjust these ranges
		upper_red = np.array([10, 255, 255]) # according to the needle color
		mask1 = cv2.inRange(hsv, lower_red, upper_red)
		lower_red = np.array([170, 50, 50])   
		upper_red = np.array([180, 255, 255])
		mask2 = cv2.inRange(hsv, lower_red, upper_red)


		mask = mask1 | mask2

		needle_angle = self.find_needle_angle(mask, visualize=visualize) # Replace with your needle angle detection logic

		if needle_angle is None: # In case an error occurs in angle finding (e.g., no lines found)
			return -1
		
		print(needle_angle)
		min_angle = 25  # Empty angle
		max_angle = 155  # Full angle
		fuel_level = (needle_angle - min_angle) / (max_angle - min_angle)
		fuel_level = np.clip(fuel_level, 0.0, 1.0)  # Ensure it's within 0-1

		print(fuel_level)

		return fuel_level
	

	def detect_gauge(self, image, method='circle_detection', template_path="gauge_template.jpg", visualize=True):
		"""
		Detects the fuel gauge in an image.

		Args:
			image (numpy.ndarray): The input image.
			method (str, optional): The method to use for gauge detection. 
				'template_matching' uses template matching (requires `template_path`).
				'circle_detection' uses Hough circle detection.
				Defaults to 'template_matching'.
			template_path (str, optional): Path to the template image if using template matching. Defaults to None.
			visualize (bool, optional): Whether to display the detected gauge. Defaults to False.

		Returns:
			tuple: Bounding box of the gauge (x, y, w, h) or None if not found.
		"""

		if method == 'template_matching':
			if template_path is None:
				raise ValueError("template_path must be provided when using template matching.")
			template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
			template = cv2.resize(template, (0, 0), fx=0.5, fy=0.5)
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)  # Use normalized cross-correlation
			_, max_val, _, max_loc = cv2.minMaxLoc(res)
			if max_val > 0.7:  # Set a threshold for a good match  # Adjust threshold as needed.
				h, w = template.shape
				x, y = max_loc
				gauge_bbox = (x, y, w, h)

			else:
				 gauge_bbox = None




		elif method == 'circle_detection': # Hough Circle Transform
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			blurred = cv2.medianBlur(gray, 5) # Blur to reduce noise
			circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20,
									   param1=50, param2=30, minRadius=0, maxRadius=0)

			if circles is not None:
				circles = np.uint16(np.around(circles))
				largest_circle = max(circles[0, :], key=lambda x: x[2])  # Find circle with largest radius
				x, y, r = np.int16(largest_circle)
				gauge_bbox = (max(x-r,0), max(y-r,0), 2*r, 2*r)

			else:
				gauge_bbox = None


		else:
			raise ValueError("Invalid method specified. Choose 'template_matching' or 'circle_detection'.")


		if visualize and gauge_bbox is not None:  # Corrected visualization logic
			x, y, w, h = gauge_bbox
			cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
			plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
			plt.title("Detected Gauge")
			plt.show()

		return gauge_bbox
	
	
	def find_needle_angle(self, mask, visualize):
		"""(Placeholder) Finds the angle of the needle."""

		edges = cv2.Canny(mask, 50, 150, apertureSize=3)

		cv2.imshow("idk", edges)
		lines = cv2.HoughLinesP(edges, 1, np.pi/220, threshold=20, minLineLength=10, maxLineGap=10)

		if lines is not None:
			longest_line = None
			max_length = 0


			for line in lines:
				x1, y1, x2, y2 = line[0]
				length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
				if length > max_length:
					max_length = length
					longest_line = line

			if longest_line is not None:

				x1, y1, x2, y2 = longest_line[0]

				angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi # Convert from radians to degrees


				if visualize:
					# Create a copy of the mask with three color channels to draw the line on it
					mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

					cv2.line(mask_color, (x1, y1), (x2, y2), (0, 0, 255), 2) # Draw on the color copy
					plt.imshow(mask_color)
					plt.title(f"Needle Angle: {angle:.2f}")
					plt.show()


				return angle
			else:
				print("Error: No line detected.")
				return None
		else:
			print("Error: No lines detected by HoughLinesP.")
			return None

	def segment(self, image, text):
		# Use previous functions to segment the image
  
		self.load_and_predict_dino(image, text)
		number_classes = self.segment_masks()
		return f'There have been segmented a total of {number_classes} classes of {text} in the image.'
	
	def segment_unique(self, image, text):
		# Use previous unique segmentation
		self.load_and_predict_dino(image, text)
		self.segment_first_mask_variations()
		return f'The object {text} has been segmented in the image following user feedback.'
	
	def change_color(self, image, color):
		# Use previous functions to change the color of the image
		self.image = image
		self.apply_masks_to_image(color)
		return f'The color of the image has been changed to {color}.'
	
	def calculate_crop_percentage(self, image):
		# Use previous functions to calculate the crop percentage
		percentage = self.analyze_crop_coverage(image, visualize=False)
		return f'The crop percentage in the image is {percentage}.'




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

	text = segmenter.transcribe_text_from_image("./best_summer_ever.png")
	print(text)
	"""
	boxes = segmenter.load_and_predict_dino("gauge.png", "gauge.")
	masks = segmenter.segment_masks()
	final_image = segmenter.apply_masks_to_image(color=(255, 0, 0))
	plt.imshow(final_image)
	plt.axis('off')
	plt.show()
	print(f"Number of masks: {segmenter.count_masks()}")
	segmenter.segment_first_mask_variations()
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
	
	image_path = "crops2.png"  
	crop_coverage = segmenter.analyze_crop_coverage(image_path)
	print(f"Estimated crop coverage: {crop_coverage:.2f}%")
	
	palette = segmenter.create_palette("faces/cai3.jpg", num_colors=5, display=True)
	
	image_path ="faces/cai.jpg"
	segmenter.remove_from_click(image_path)
	
	boxes = segmenter.load_and_predict_dino("car_parking.png", "car.")
	masks = segmenter.segment_masks()
	print(f"Number of masks: {segmenter.count_masks()}")


	plt.imshow(segmenter.get_segmented_image())


	color = segmenter.count_masks_by_color((150,150,150))
	print(f"Number of white masks: {color}")
	"""