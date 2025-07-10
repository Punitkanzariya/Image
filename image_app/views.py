import os
import gc
from django.shortcuts import render
from django.conf import settings
from PIL import Image
import numpy as np
from .realesrgan_model import upsampler  # ⬅️ Imported globally loaded model

def upscale_view(request):
    if request.method == 'POST':
        image = request.FILES['image']

        # Convert PIL image
        img = Image.open(image).convert('RGB')
        img_np = np.array(img)

        # === Enhancement using globally loaded upsampler ===
        output_np, _ = upsampler.enhance(img_np)
        output_img = Image.fromarray(output_np)

        # Save output image
        result_dir = os.path.join(settings.MEDIA_ROOT, 'results')
        os.makedirs(result_dir, exist_ok=True)
        output_path = os.path.join(result_dir, 'result.png')
        output_img.save(output_path)

        # Save uploaded input too (optional)
        upload_path = os.path.join(settings.MEDIA_ROOT, 'uploads', image.name)
        os.makedirs(os.path.dirname(upload_path), exist_ok=True)
        img.save(upload_path)

        # Clean up memory
        del img, img_np, output_np, output_img
        gc.collect()

        return render(request, 'result.html', {
            'input_url': f"/media/uploads/{image.name}",
            'output_url': '/media/results/result.png'
        })

    return render(request, 'index.html')
