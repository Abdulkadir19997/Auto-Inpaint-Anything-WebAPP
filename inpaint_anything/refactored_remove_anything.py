import torch


# from sam_segment import predict_masks_with_sam
from inpaint_anything.lama_inpaint import inpaint_img_with_lama


class RemoveAnything:
    def __init__(self, device=None):
        # Default parameters
        self.lama_config = "inpaint_anything/lama/configs/prediction/default.yaml"
        self.lama_ckpt = "inpaint_anything/pretrained_models/big-lama"
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    def remove_anything(self, img, mask):
        
        # Inpaint the masked image
        img_inpainted = inpaint_img_with_lama(
            img, mask, self.lama_config, self.lama_ckpt, 8, self.device)

        return img_inpainted

# Usage
# inpainter = RemoveAnything()
# inpainter.remove_anything()
