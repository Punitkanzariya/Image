import io
import base64
import gc
from django.shortcuts import render
from PIL import Image
import numpy as np
from .realesrgan_model import upsampler  # Preloaded model

def upscale_view(request):
    if request.method == 'POST':
        image = request.FILES['image']

        # Convert PIL image
        img = Image.open(image).convert('RGB')
        img_np = np.array(img)

        # Enhance image using Real-ESRGAN
        output_np, _ = upsampler.enhance(img_np)
        output_img = Image.fromarray(output_np)

        # Convert input and output images to base64 (for direct embedding)
        def image_to_base64(pil_image):
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

        input_b64 = image_to_base64(img)
        output_b64 = image_to_base64(output_img)

        # Clean up memory
        del img, img_np, output_np, output_img
        gc.collect()

        return render(request, 'result.html', {
            'input_b64': input_b64,
            'output_b64': output_b64
        })

    return render(request, 'index.html')
