import os
import torch
from django.conf import settings
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# Default values
DEFAULT_MODEL_NAME = 'RealESRGAN_x4plus'
DEFAULT_SCALE = 4
DEFAULT_TILE = 0
DEFAULT_TILE_PAD = 10
USE_HALF = torch.cuda.is_available()

# Load model path
model_path = os.path.join(settings.BASE_DIR, 'weights', f'{DEFAULT_MODEL_NAME}.pth')

# Initialize model architecture
model = RRDBNet(
    num_in_ch=3, num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=DEFAULT_SCALE
)

# Initialize RealESRGANer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
upsampler = RealESRGANer(
    scale=DEFAULT_SCALE,
    model_path=model_path,
    model=model,
    tile=DEFAULT_TILE,
    tile_pad=DEFAULT_TILE_PAD,
    pre_pad=0,
    half=USE_HALF,
    device=device
)
