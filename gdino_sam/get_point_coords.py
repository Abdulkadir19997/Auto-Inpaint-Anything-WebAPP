import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
# import sys
# sys.path.append("/home/src-01/Documents/projects/personal/Automated-Inpaint-Anything")
from config import settings

class GroundingDINO:
    def __init__(self, model_name="IDEA-Research/grounding-dino-base"):
        self.device = settings.DEVICE
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(self.device)
    
    def bbox_detector(self, image, prompt):
        try:
            image = image.convert("RGB")
            text = prompt.lower().strip() + "."
            inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.4,
                text_threshold=0.3,
                target_sizes=[image.size[::-1]]
            )
            point_coords = []
            for result in results:
                boxes = result["boxes"].cpu().detach().numpy()
                labels = result["labels"]
                # Calculate center points for each box
                for box in boxes:
                    x_center = (box[0] + box[2]) / 2
                    y_center = (box[1] + box[3]) / 2
                    point_coords.append([x_center, y_center])
            return [point_coords[0]], labels, boxes
        except ValueError as e:
            print(f"Error processing image: {str(e)}")
            return [], []  # Or handle the error in a way that fits your application

# Example usage:
# detector = GroundingDINO()

# image_path = "inpaint_anything/example/fill-anything/sample1.png"
# prompt = "dog."

# point_coords, labels = detector.bbox_detector(image_path, prompt)

# print(f"Center Coordinates of Bounding Boxes: {point_coords}", labels)
