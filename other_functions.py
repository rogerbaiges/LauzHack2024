import cv2
from matplotlib import pyplot as plt
from segmentor import ImageSegmenter

olives = 'satelite.png'

segmenter = ImageSegmenter(
	grounding_dino_cfg="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
	grounding_dino_weights="GroundingDINO/weights/groundingdino_swint_ogc.pth",
	sam_cfg="C:/Users/Cai Selvas Sala/GIA_UPC/Personal/LauzHack/LauzHack_2024/LauzHack2024/samsam/sam2/configs/sam2.1/sam2.1_hiera_t.yaml",
	sam_weights="C:/Users/Cai Selvas Sala/GIA_UPC/Personal/LauzHack/LauzHack_2024/LauzHack2024/samsam/weights/sam2.1_hiera_tiny.pt"
)

segmenter.find_similar_objects(olives)