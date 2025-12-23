import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "AI Fast Image Enhancer Online"}

# ULTRA LITE CONFIGURATION
# Scale=2 and Tile=120 will make it much faster and prevent "Killed" error
model_resgrgan = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
upsampler = RealESRGANer(
    scale=2, 
    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
    model=model_resgrgan,
    tile=120, 
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
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Fast Resizing: Downscale if image is too large to save RAM
    h, w = img.shape[:2]
    if h > 800 or w > 800:
        img = cv2.resize(img, (800, int(800 * h / w)), interpolation=cv2.INTER_AREA)

    # Process
    _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)

    output_path = "result.png"
    cv2.imwrite(output_path, output)
    return FileResponse(output_path)
EOF
