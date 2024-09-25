from gdino_sam.get_segmentation_masks import SegmentImages
from inpaint_anything.stable_diffusion_inpaint import fill_img_with_sd
from inpaint_anything.utils.utils import dilate_mask, load_imgs_to_array
from config import settings

class FillAnything:

    def fill_anything(self, img, point_coords, text_prompt, bbox):
        try:
            device = settings.DEVICE   
                        # Convert img and mask to numpy arrays
            segmentor = SegmentImages()
            mask = segmentor.get_segmentation(img, point_coords, bbox)
            print("segmentation passed")
            
            img = load_imgs_to_array(img)
            # Check if img and mask are valid numpy arrays
            if img is None or mask is None:
                raise ValueError("Image or mask is None or could not be loaded.")
            # Dilate mask depending on the circumsatance

            if settings.DILATE_KERNEL_SIZE_REPLACE is not None:
                mask = dilate_mask(mask, settings.DILATE_KERNEL_SIZE_FILL)

            img_filled = fill_img_with_sd(
                img, mask, text_prompt, device=device)

            return img_filled
        except Exception as e:
            print(f"Error during filling: {str(e)}")
            return None
# inpainter = ImageInpainter()
# inpainter.process_image()