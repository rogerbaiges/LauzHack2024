import cv2
import numpy as np
import sys
import os
import matplotlib.pyplot as plt  # For visualization (optional)
from hydra import initialize_config_module
from hydra.core.global_hydra import GlobalHydra
if not GlobalHydra.instance().is_initialized():
    initialize_config_module("samsam/sam2", version_base="1.2")
# SAM2 imports (make sure the path is correct)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, './samsam/'))) # Update path if necessary
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam_cfg= "C:/Users/Usuario/Documents/Projectes/LauzHack2024/LauzHack2024/samsam/sam2/configs/sam2.1/sam2.1_hiera_t.yaml"
sam_weights="C:/Users/Usuario/Documents/Projectes/LauzHack2024/LauzHack2024/samsam/weights/sam2.1_hiera_tiny.pt"

def get_user_click(image):
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



def extract_mask_with_sam2(image, click_point):
    """Extracts a mask using SAM2 based on the user's click."""

    print(sam_cfg)
    print(sam_weights)
    sam2 = build_sam2(sam_cfg, sam_weights, device="cpu")  # Build the SAM2 model
    predictor = SAM2ImagePredictor(sam2)
    predictor.set_image(image) 
    masks, _, _ = predictor.predict(
        point_coords=np.array([click_point]),  # Click coordinates as a NumPy array
        point_labels=np.array([1]),              # Positive label (1 for foreground)
        multimask_output=False,
    )


    plt.imshow(masks[0])
    plt.title("Extracted Mask")
    plt.show()

    return masks[0]  # Return the first mask (since multimask_output=False)


def find_similar_features_in_image(image, mask, size_tolerance=0.8, rotation_tolerance=90):
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


def remove_and_inpaint_cv2(image, mask):
    """Removes the masked region and inpaints it using OpenCV's inpainting methods."""

    mask = mask.astype(np.uint8) * 255  # Ensure mask is 8-bit
    image = image.astype(np.uint8)  # Image should also be 8-bit

    # OpenCV inpainting (choose one of the methods below)
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=0.5, flags=cv2.INPAINT_TELEA) # Telea
    # or
    # inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)  # Navier-Stokes


    return inpainted_image







# Example usage:
image_path = "faces/cai.jpg"
image = cv2.imread(image_path)

click_point = get_user_click(image)
if click_point is None:
    print("No click received. Exiting.")
    sys.exit()

mask = extract_mask_with_sam2(image, click_point)


#--- Visualization of the mask (optional) ---
# plt.imshow(mask)
# plt.title("Extracted Mask")
# plt.show()
#--------------------------------------------



inpainted_image = remove_and_inpaint_cv2(image.copy(), mask)



# --- Visualization ---
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) #Original image
plt.title("Original Image with Mask")
plt.subplot(1, 2, 2)
plt.imshow(inpainted_image)
plt.title("Inpainted Image")
plt.show()


tolerance = 5 # Adjust as needed.
similar_regions = find_similar_features_in_image(image, mask, tolerance)



if similar_regions:
    print("Similar features found:")
    for x, y, w, h, _ in similar_regions:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Detected Features", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No similar features found.")