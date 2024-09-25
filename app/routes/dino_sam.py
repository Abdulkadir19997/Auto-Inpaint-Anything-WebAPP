from fastapi import UploadFile, File, Form, APIRouter, HTTPException
from PIL import Image
import io
import base64
from app.schemas.response_bodies import SegmentationResponse
from gdino_sam.get_point_coords import GroundingDINO
from gdino_sam.get_segmentation_masks import SegmentImages

# Initialize FastAPI router
router = APIRouter()

# Initialize the GroundingDINO and SegmentImages
detector = GroundingDINO()
segmentor = SegmentImages()

# Endpoint to accept image and text
@router.post("", response_model=SegmentationResponse)
async def upload_image_text(image: UploadFile = File(...), text: str = Form(...)):
    try:
        # Read the uploaded image
        image_data = await image.read()
        
        # Convert image bytes to PIL Image for processing
        img = Image.open(io.BytesIO(image_data))

        # Process the image with GroundingDINO and SegmentImages
        point_coords, labels, boxes = detector.bbox_detector(img, text)
        print(point_coords)
        print(labels)
        print(img)
        print(boxes)
        print("object detection passed")
        segmented_result = segmentor.get_segmentation(img, point_coords, boxes)
        print("segmentation passed")
        # Convert the segmented result to bytes without saving to disk
        mask_image = Image.fromarray(segmented_result)  
        
        # Convert the PIL Image to bytes (PNG format)
        img_byte_arr = io.BytesIO()
        mask_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Encode the segmented image in base64
        encoded_image = base64.b64encode(img_byte_arr).decode('utf-8')
        print(image.filename)
        # Return response body
        return SegmentationResponse(
            point_coords=point_coords,
            bbox_coords=boxes,
            labels=labels,
            segmented_result=encoded_image,
            image_name=image.filename  # Return the original image name
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the image: {str(e)}")
