import numpy as np
import torch
from gdino_sam.get_segmentation_masks import SegmentImages
from inpaint_anything.stable_diffusion_inpaint import replace_img_with_sd
from inpaint_anything.utils.utils import dilate_mask, load_imgs_to_array
from config import settings

class ReplaceAnything:

    def replace_anything(self, img, point_coords, text_prompt, bbox_coords):
        try:
            device = settings.DEVICE   
                        # Convert img and mask to numpy arrays
            segmentor = SegmentImages()
            mask = segmentor.get_segmentation(img, point_coords, bbox_coords)
            print("segmentation passed")
            
            img = load_imgs_to_array(img)
            # Check if img and mask are valid numpy arrays
            if img is None or mask is None:
                raise ValueError("Image or mask is None or could not be loaded.")
            
            # Dilate mask depending on the circumsatance
            if settings.DILATE_KERNEL_SIZE_FILL is not None:
                mask = dilate_mask(mask, 1)

            img_replaced = replace_img_with_sd(
                img, mask, text_prompt, device=device)
            return img_replaced

        except Exception as e:
            print(f"Error during replacing: {str(e)}")
            return None
