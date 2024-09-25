from pathlib import Path
import torch
import numpy as np
from inpaint_anything.sam_segment import predict_masks_with_sam
from inpaint_anything.utils import get_clicked_point
from config import settings

class SegmentImages:
    def __init__(self):
        self.coords_type = "key_in"  # Assuming you still want to use this type
        self.output_dir = "./results"  # Output directory for saving masks (if necessary)

    def load_img_to_array(self, img):
        """Convert a PIL Image to a numpy array."""
        if img.mode == "RGBA":
            img = img.convert("RGB")  # Convert RGBA images to RGB
        return np.array(img)
    
    def get_segmentation(self, img, point_coords, bbox):
        try:
            # Use provided coordinates or generate based on click behavior
            if self.coords_type == "click":
                latest_coords = get_clicked_point(img)  # Updated to use the image directly
            else:
                latest_coords = point_coords

            # Convert the PIL image to numpy array
            img_array = self.load_img_to_array(img)
            
            # Predict masks using SAM
            masks, _, _ = predict_masks_with_sam(
                img_array,  # Image is directly passed in as a numpy array
                latest_coords,
                [1],
                model_type=settings.SAM_MODEL_TYPE,
                ckpt_p=settings.SAM_CHECKPOINT_PATH,
                device=settings.DEVICE,
                bbox=bbox
            )
            masks = masks.astype(np.uint8) * 255  # Convert masks

            # Visualize and save results
            # img_stem = Path(self.input_img).stem
            # out_dir = Path(settings.OUTPUT_DIR) / img_stem
            # out_dir.mkdir(parents=True, exist_ok=True)
            # filler = FillAnything()
            # image_filled = filler.fill_anything(img, masks[1])
            # replacer = ReplaceAnything()
            # img_replaced = replacer.replace_anything(img, masks[1])
            # remover = RemoveAnything()
            # img_removed = remover.remove_anything(img, masks[1])
            # save_masks(self.point_coords, out_dir, img, masks[1], self.point_labels, image_filled, settings.ENDPOINT_NAME_FILL)
            return masks[1]
        except Exception as e:
            print(f"Error during segmentation: {str(e)}")
            return None

# segmentor = SegmentImages()
# segmentor.get_segmentation([722.8389892578125, 591.0413818359375], [1])