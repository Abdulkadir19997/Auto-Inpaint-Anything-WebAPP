from fastapi import UploadFile, File, Form, APIRouter, HTTPException
from PIL import Image
import io
import base64
import json

import numpy as np
from app.ai_services.get_fill_anything_result import FillAnything
from app.schemas.response_bodies import InpaintAnythingResponse
from typing import List

# Initialize FastAPI router
router = APIRouter()

# Initialize the RemoveAnything service
filler = FillAnything()

# Endpoint to accept image and point coordinates
@router.post("", response_model=InpaintAnythingResponse)
async def upload_image_text(
    original_image: UploadFile = File(...), 
    point_coords: str = Form(...),
    bbox_coords: str = Form(...),
    text_prompt: str = Form(...)
):
    try:
        # Read the uploaded image
        original_image_data = await original_image.read()
        
        # Convert image bytes to PIL Image for processing
        img = Image.open(io.BytesIO(original_image_data))

        # Parse the point_coords from string (Form data) to a list of lists of floats
        point_coords_list: List[List[float]] = json.loads(point_coords)
        bbox_coords_list: List[List[float]] = json.loads(bbox_coords)
        # Validate the point_coords_list format (each coordinate should be a list of two floats)
        if not isinstance(point_coords_list, list) or not all(isinstance(coord, list) and len(coord) == 2 for coord in point_coords_list):
            raise HTTPException(status_code=400, detail="Invalid format for point_coords")
        
        # Input the original image and the point coordinates into remove_anything
        filled_result = filler.fill_anything(img, point_coords_list, text_prompt, np.array(bbox_coords_list))

        # Convert the removed result (numpy array) to a PIL Image    
        # Data type conversion
        if filled_result.dtype != np.uint8:
            if filled_result.max() <= 1.0:  # Assuming the data is normalized to 0-1
                filled_result = (filled_result * 255).astype(np.uint8)
            else:
                filled_result = filled_result.astype(np.uint8)

        filled_result = Image.fromarray(filled_result)
        
        # Convert the PIL Image to bytes (PNG format)
        img_byte_arr = io.BytesIO()
        filled_result.save(img_byte_arr, format='PNG')

        img_byte_arr = img_byte_arr.getvalue()
        # Encode the result image in base64
        encoded_image = base64.b64encode(img_byte_arr).decode('utf-8')

        # Return response body
        return InpaintAnythingResponse(
            image_result=encoded_image,
            image_name=original_image.filename  # Return the original image name
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the image: {str(e)}")
