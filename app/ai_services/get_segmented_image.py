# Example usage:
from gdino_sam.get_point_coords import GroundingDINO
from gdino_sam.get_segmentation_masks import SegmentImages



detector = GroundingDINO()
segmentor = SegmentImages()


image_path = "inpaint_anything/example/fill-anything/sample1.png"
prompt = "dog."

def get_segmented_result():
    
    point_coords, labels = detector.bbox_detector(image_path, prompt)
    print(f"Center Coordinates of Bounding Boxes: {point_coords}", labels)
    segmented_result = segmentor.get_segmentation(point_coords, labels)
    return point_coords, labels, segmented_result