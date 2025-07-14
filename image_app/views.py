import io
import base64
import gc
from django.shortcuts import render
from django.http import HttpResponseBadRequest
from PIL import Image
import numpy as np
from .realesrgan_model import upsampler  # Preloaded model

MAX_IMAGE_SIZE_MB = 5  # You can adjust this limit

def upscale_view(request):
    if request.method == 'POST':
        image = request.FILES.get('image')

        if not image:
            return HttpResponseBadRequest("No image uploaded.")

        # Size check
        if image.size > MAX_IMAGE_SIZE_MB * 1024 * 1024:
            return HttpResponseBadRequest("Image too large. Max 5MB allowed.")

        try:
            # Convert to RGB
            img = Image.open(image).convert('RGB')

            # Optional resize to prevent memory issues
            MAX_DIMENSION = 2000  # pixels
            if max(img.size) > MAX_DIMENSION:
                img.thumbnail((MAX_DIMENSION, MAX_DIMENSION))

            img_np = np.array(img)

            # Real-ESRGAN enhancement
            output_np, _ = upsampler.enhance(img_np)
            output_img = Image.fromarray(output_np)

            # Encode to base64
            def image_to_base64(pil_image):
                buffer = io.BytesIO()
                pil_image.save(buffer, format='PNG')
                return base64.b64encode(buffer.getvalue()).decode('utf-8')

            input_b64 = image_to_base64(img)
            output_b64 = image_to_base64(output_img)

            # Cleanup
            del img, img_np, output_np, output_img
            gc.collect()

            return render(request, 'result.html', {
                'input_b64': input_b64,
                'output_b64': output_b64
            })

        except Exception as e:
            return HttpResponseBadRequest(f"Error during upscaling: {str(e)}")

    return render(request, 'index.html')
