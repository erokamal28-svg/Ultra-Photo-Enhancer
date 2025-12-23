import os
import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

app = FastAPI()

# Enable CORS for frontend connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root route to check if server is online
@app.get("/")
async def root():
    return {"status": "AI Fast Image Enhancer Online"}

# Initialize Fast AI Models (Optimized for CPU)
# Using Scale 2 to prevent memory crashes (Killed Error)
model_resgrgan = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
upsampler = RealESRGANer(
    scale=2, 
    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
    model=model_resgrgan,
    tile=100, 
    tile_pad=10,
    pre_pad=0,
    half=False
)

face_enhancer = GFPGANer(
    model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
    upscale=2,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=upsampler
)

@app.post("/enhance")
async def enhance_image(file: UploadFile = File(...)):
    # Read uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Automatic resizing for large images to prevent "Killed" error
    h, w = img.shape[:2]
    if h > 1200 or w > 1200:
        img = cv2.resize(img, (w//2, h//2), interpolation=cv2.INTER_AREA)

    # Fast AI Enhancement Process
    _, _, output = face_enhancer.enhance(
        img, 
        has_aligned=False, 
        only_center_face=False, 
        paste_back=True
    )

    output_path = "enhanced_result.png"
    cv2.imwrite(output_path, output)
    
    return FileResponse(output_path)
