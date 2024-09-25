from pydantic import BaseModel
from typing import List

# Pydantic model for the response body
class SegmentationResponse(BaseModel):
    point_coords: List[List[float]]
    bbox_coords: List[List[float]]
    labels: List[str]
    segmented_result: str  # Base64 encoded image string
    image_name: str  # Return the original image name


class InpaintAnythingResponse(BaseModel):
    image_result: str  # Base64 encoded image string
    image_name: str
