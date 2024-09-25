import torch
from inpaint_anything.stable_diffusion_inpaint import replace_img_with_sd
from config import settings
from inpaint_anything.utils.utils import dilate_mask

class ReplaceAnything:
    def __init__(self):
        # Default values are set as attributes
        self.text_prompt = "sit on the swing"


    def replace_anything(self, img, mask):
        device = settings.DEVICE   
             
        # Dilate mask depending on the circumsatance
        if settings.DILATE_KERNEL_SIZE_REPLACE is not None:
            mask = dilate_mask(mask, settings.DILATE_KERNEL_SIZE_REPLACE)

        img_replaced = replace_img_with_sd(
            img, mask, self.text_prompt, device=device)
        return img_replaced

