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
    return {"status": "AI Image Enhancer Server is Online"}

# Initialize AI Models (Real-ESRGAN + GFPGAN)
model_resgrgan = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4,
    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    model=model_resgrgan,
    tile=400,
    tile_pad=10,
    pre_pad=0,
    half=True if torch.cuda.is_available() else False
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
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Professional Enhancement (Face + 8K Texture)
    _, _, output = face_enhancer.enhance(
        img, 
        has_aligned=False, 
        only_center_face=False, 
        paste_back=True
    )

    output_path = "enhanced_output.png"
    cv2.imwrite(output_path, output)
    
    return FileResponse(output_path)
