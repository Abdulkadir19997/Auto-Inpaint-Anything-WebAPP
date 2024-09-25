#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
import torch


class Settings(BaseSettings):


    #######
    # AI configurations part
    # BASE_URI : str = "http://app.searcly.com:5004"


    # Segmentation parameters
    DILATE_KERNEL_SIZE_REMOVE : int = 15 # if remove
    DILATE_KERNEL_SIZE_REPLACE : Optional[int] = None # if replace
    DILATE_KERNEL_SIZE_FILL : int = 50 # if fill
    SAM_MODEL_TYPE : str = "vit_h"


    # Save paths depending on endpoint name
    ENDPOINT_NAME_REMOVE : str = "remove_"
    ENDPOINT_NAME_REPLACE : str = "replace_"
    ENDPOINT_NAME_FILL : str = "fill_"
 

    SAM_CHECKPOINT_PATH : str = "inpaint_anything/pretrained_models/sam_vit_h_4b8939.pth"

    # Choose device
    DEVICE : str = "cuda" if torch.cuda.is_available() else "cpu"

    # Save path
    OUTPUT_DIR : str = "results"
    

    class Config:
        case_sensitive = True
        env_file = '.env'
        


settings = Settings()
