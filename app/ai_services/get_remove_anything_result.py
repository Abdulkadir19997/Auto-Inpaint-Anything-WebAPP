import torch
from gdino_sam.get_segmentation_masks import SegmentImages
from inpaint_anything.lama_inpaint import inpaint_img_with_lama
from inpaint_anything.utils.utils import dilate_mask, load_imgs_to_array
from config import settings


class RemoveAnything:
    def __init__(self, device=None):
        # Default parameters
        self.lama_config = "inpaint_anything/lama/configs/prediction/default.yaml"
        self.lama_ckpt = "inpaint_anything/pretrained_models/big-lama"
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    def remove_anything(self, img, point_coords, bbox):
        try:
            # Convert img and mask to numpy arrays
            segmentor = SegmentImages()
            mask = segmentor.get_segmentation(img, point_coords, bbox)
            print("segmentation passed")
            
            img = load_imgs_to_array(img)
            # Check if img and mask are valid numpy arrays
            if img is None or mask is None:
                raise ValueError("Image or mask is None or could not be loaded.")
        
            mask = dilate_mask(mask, settings.DILATE_KERNEL_SIZE_REMOVE)

            #Remove the wanted object from image using LaMa
            removed_image = inpaint_img_with_lama(
                img, mask, self.lama_config, self.lama_ckpt, 8, self.device
            )

            return removed_image

        except Exception as e:
            print(f"Error during removal: {str(e)}")
            return None