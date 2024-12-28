from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch
import os

# Configuration
MODEL_NAME = "CompVis/stable-diffusion-v1-4"
OUTPUT_DIR = "./dreambooth-finetuned"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT = "a [V] wizard in a field"

# Load the base Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained(MODEL_NAME, cache_dir='/w/331/usen/cache/huggingface_cache')

# Load the fine-tuned UNet
pipe.unet = UNet2DConditionModel.from_pretrained(os.path.join(OUTPUT_DIR, "unet"))

# Load the fine-tuned text encoder
pipe.text_encoder = CLIPTextModel.from_pretrained(os.path.join(OUTPUT_DIR, "text_encoder"))

# Move pipeline to the appropriate device
pipe.to(DEVICE)

# Generate an image
with torch.no_grad():
    generated_image = pipe(PROMPT).images[0]

# Save the generated image
generated_image.save(f"{OUTPUT_DIR}/generated_image.png")
print("Image generated and saved.")
