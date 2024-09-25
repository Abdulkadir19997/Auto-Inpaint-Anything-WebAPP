import torch
from inpaint_anything.stable_diffusion_inpaint import fill_img_with_sd

class FillAnything:
    def __init__(self):
        self.text_prompt = "a teddy bear on a bench"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fill_anything(self, img, mask):
        img_filled = fill_img_with_sd(
            img, mask, self.text_prompt, device=self.device)
        # img_filled_p = out_dir / f"filled_{idx}.png"
        # save_array_to_img(img_filled, img_filled_p)
        return img_filled

# inpainter = ImageInpainter()
# inpainter.process_image()