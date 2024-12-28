# Import libraries
import os
import sys
import random
import types
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler
from torch.amp import autocast, GradScaler
from diffusers.training_utils import compute_snr
from transformers import CLIPTokenizer
from torch.optim import AdamW
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import itertools
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed




accelerator = Accelerator(
    mixed_precision="fp16",  # or "bf16" if supported
    gradient_accumulation_steps=5,  # Adjust based on your memory constraints
    cpu=False  # Set to True if you want to run on CPU
)

set_seed(42)

# Configuration
MODEL_NAME = "CompVis/stable-diffusion-v1-4"
INSTANCE_PROMPT = "a [V] wizard"
CLASS_PROMPT = "wizard"
LEARNING_RATE = 5e-6
TRAIN_EPOCHS = 500
BATCH_SIZE = 1
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "./dreambooth-finetuned"
IMAGES_DIR = "./images"
MASKS_DIR = "./masks"
CLASS_IMAGES_DIR = "./class_images"
NUM_CLASS_IMAGES = 100
PRIOR_LOSS_WEIGHT = 1.0
LOCALIZATION_WEIGHT = 1e-3

# Initialize tokenizer and model
tokenizer = CLIPTokenizer.from_pretrained(
    "openai/clip-vit-large-patch14",
    cache_dir='/w/331/usen/cache/huggingface_cache'
)

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    cache_dir='/w/331/usen/cache/huggingface_cache'
)

pipe.to(accelerator.device)

# Add custom token for the subject
new_token = "[V]"
num_added_tokens = tokenizer.add_tokens(new_token)
if num_added_tokens > 0:
    pipe.text_encoder.resize_token_embeddings(len(tokenizer))

# Enable training for UNet and text encoder
for param in pipe.unet.parameters():
    param.requires_grad = True
for param in pipe.text_encoder.parameters():
    param.requires_grad = True
for param in pipe.vae.parameters():
    param.requires_grad = False  # VAE parameters are frozen


# Store cross-attention scores
def store_cross_attention_scores(unet):
    from diffusers.models.attention_processor import Attention, AttnProcessor, AttnProcessor2_0

    attention_scores = {}

    def make_attention_score_hook(name):
        def hook(module, query, key, attention_mask=None):
            attention_probs = module.old_get_attention_scores(query, key, attention_mask)
            attention_scores[name] = attention_probs  # Do not detach
            return attention_probs

        return hook

    for name, module in unet.named_modules():
        if isinstance(module, Attention) and "attn2" in name:
            if isinstance(module.processor, AttnProcessor2_0):
                module.set_processor(AttnProcessor())
            module.old_get_attention_scores = module.get_attention_scores
            module.get_attention_scores = types.MethodType(make_attention_score_hook(name), module)

    unet.attention_scores = attention_scores  # Store attention_scores in the unet

    return unet


pipe.unet = store_cross_attention_scores(pipe.unet)


# Balanced L1 Loss for localization
class BalancedL1Loss(nn.Module):
    def __init__(self, threshold=1.0, normalize=True):
        super().__init__()
        self.threshold = threshold
        self.normalize = normalize

    def forward(self, attention_scores, segmentation_masks):
        bsnh, spatial_res, num_tokens = attention_scores.shape

        num_heads = attention_scores.size(0) // segmentation_masks.size(0)
        batch_size = segmentation_masks.size(0)

        attention_scores = attention_scores.view(batch_size, num_heads, spatial_res, num_tokens)

        segmentation_masks = F.interpolate(
            segmentation_masks, size=(spatial_res, num_tokens), mode="bicubic", align_corners=False
        )

        segmentation_masks = segmentation_masks.unsqueeze(1)

        if self.normalize:
            attention_scores = attention_scores / (attention_scores.max(dim=3, keepdim=True)[0] + 1e-3)

        background_masks = 1 - segmentation_masks
        background_loss = (attention_scores * background_masks).sum(dim=3) / (background_masks.sum(dim=3) + 1e-3)
        object_loss = (attention_scores * segmentation_masks).sum(dim=3) / (segmentation_masks.sum(dim=3) + 1e-3)

        return background_loss.mean() - object_loss.mean()


localization_loss_fn = BalancedL1Loss(normalize=True)


# Load and preprocess data
def load_data(images_dir, masks_dir):
    images = []
    masks = []
    file_names = sorted(os.listdir(images_dir))
    for file_name in file_names:
        img_path = os.path.join(images_dir, file_name)
        mask_path = os.path.join(masks_dir, file_name.replace(".jpeg", ".npy"))
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = np.load(mask_path)
        images.append(img)
        masks.append(mask)
    return images, masks


def preprocess_data(images, masks):
    preprocessed_images = []
    preprocessed_masks = []
    for img, mask in zip(images, masks):
        img = cv2.resize(img, (256, 256))
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).to(torch.float32) / 255.0  # Normalize image tensor
        preprocessed_images.append(img_tensor)

        mask = cv2.resize(mask, (64, 64), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(torch.float32)
        preprocessed_masks.append(mask_tensor)

    return preprocessed_images, preprocessed_masks


def preprocess_images(images):
    preprocessed_images = []
    for img in images:
        img = img.resize((256, 256))
        img_tensor = transforms.ToTensor()(img).to(torch.float32)
        preprocessed_images.append(img_tensor)
    return preprocessed_images


# Define custom dataset
class DreamBoothDataset(Dataset):
    def __init__(self, images, masks, prompts):
        self.images = images
        self.masks = masks
        self.prompts = prompts
        self.transform = transforms.Compose([
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        # image = self.transform(image)
        mask = self.masks[idx]
        prompt = self.prompts[idx]
        return {
            'image': image,
            'mask': mask,
            'prompt': prompt
        }


# Custom collate function to handle None in 'mask'
def custom_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    prompts = [item['prompt'] for item in batch]
    if batch[0]['mask'] is not None:
        masks = torch.stack([item['mask'] for item in batch])
    else:
        masks = None
    return {'image': images, 'mask': masks, 'prompt': prompts}


# Generate class images if needed
def generate_class_images(prompt, num_images, pipeline, save_dir):
    pipeline.to(accelerator.device)
    class_images = []
    for idx in tqdm(range(num_images), desc='Generating Class Images', file=sys.stdout):
        with torch.no_grad():
            image = pipeline(prompt).images[0]
        class_images.append(image)
        image.save(os.path.join(save_dir, f"class_image_{idx}.png"))
    return class_images


# Load class images from directory
def load_class_images(class_images_dir):
    class_images = []
    file_names = sorted(os.listdir(class_images_dir))
    for file_name in file_names:
        img_path = os.path.join(class_images_dir, file_name)
        image = Image.open(img_path).convert('RGB')
        class_images.append(image)
    return class_images


# Prepare class images
def prepare_class_images(pipeline):
    if not os.path.exists(CLASS_IMAGES_DIR):
        os.makedirs(CLASS_IMAGES_DIR, exist_ok=True)
        print("Generating class images...")
        class_images = generate_class_images(CLASS_PROMPT, NUM_CLASS_IMAGES, pipeline, CLASS_IMAGES_DIR)
    else:
        print("Loading existing class images...")
        class_images = load_class_images(CLASS_IMAGES_DIR)

    class_prompts = [CLASS_PROMPT] * len(class_images)
    return class_images, class_prompts


# Training loop
def train_loop(epochs, noise_scheduler, optimizer, train_dataloader, class_dataloader):
    # scaler = GradScaler('cuda')
    pipe.unet.train()
    pipe.text_encoder.train()

    class_dataloader_iter = itertools.cycle(class_dataloader)

    for epoch in tqdm(range(epochs), desc='Training', file=sys.stdout):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_loss = 0

        for instance_batch in train_dataloader:
            optimizer.zero_grad()

            # with autocast('cuda'):

            # Instance batch
            instance_images = instance_batch['image']
            instance_masks = instance_batch['mask']
            instance_prompts = instance_batch['prompt']

            # Encode instance images to latents
            with torch.no_grad():
                instance_images = instance_images.to(accelerator.device)
                instance_latents = pipe.vae.encode(instance_images).latent_dist.sample().detach()
                instance_latents = instance_latents * 0.18215  # VAE scaling factor

            # Prepare text embeddings
            instance_text_input = tokenizer(
                instance_prompts,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(accelerator.device)
            instance_text_embeddings = pipe.text_encoder(instance_text_input.input_ids)[0]

            # Add noise
            noise = torch.randn_like(instance_latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (instance_latents.size(0),), device=accelerator.device
            ).long()
            noisy_latents = noise_scheduler.add_noise(instance_latents, noise, timesteps)

            # Reset attention_scores
            pipe.unet.attention_scores.clear()

            # UNet forward pass for instance images
            noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=instance_text_embeddings).sample

            # Compute denoising loss with SNR weighting
            snr = compute_snr(noise_scheduler, timesteps)
            base_weight = (
                    torch.stack([snr, 5.0 * torch.ones_like(snr)], dim=1).min(dim=1)[0] / snr
            )
            mse_loss_weight = base_weight
            denoise_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction='none')
            denoise_loss = denoise_loss.mean(dim=list(range(1, len(denoise_loss.shape)))) * mse_loss_weight
            denoise_loss = denoise_loss.mean()

            # Localization Loss
            if instance_masks is not None:
                resized_masks = F.interpolate(instance_masks.to(accelerator.device), size=noise_pred.shape[-2:],
                                              mode="bicubic",
                                              align_corners=False)
                localization_loss = 0
                num_layers = len(pipe.unet.attention_scores)
                for layer_scores in pipe.unet.attention_scores.values():
                    localization_loss += localization_loss_fn(layer_scores, resized_masks)
                localization_loss /= num_layers  # Average over layers
            else:
                localization_loss = 0

            # Total instance loss
            total_instance_loss = denoise_loss + LOCALIZATION_WEIGHT * localization_loss

            # Process class images
            class_batch = next(class_dataloader_iter)
            class_images = class_batch['image']
            class_prompts = class_batch['prompt']

            with torch.no_grad():
                class_images = class_images.to(accelerator.device)
                class_latents = pipe.vae.encode(class_images).latent_dist.sample().detach()
                class_latents = class_latents * 0.18215

            # Prepare class text embeddings
            class_text_input = tokenizer(
                class_prompts,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(accelerator.device)
            class_text_embeddings = pipe.text_encoder(class_text_input.input_ids)[0]

            # Add noise to class latents
            class_noise = torch.randn_like(class_latents)
            class_timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (class_latents.size(0),), device=accelerator.device
            ).long()
            noisy_class_latents = noise_scheduler.add_noise(class_latents, class_noise, class_timesteps)

            # UNet forward pass for class images
            class_noise_pred = pipe.unet(noisy_class_latents, class_timesteps,
                                         encoder_hidden_states=class_text_embeddings).sample

            # Compute prior loss
            prior_loss = F.mse_loss(class_noise_pred.float(), class_noise.float(), reduction='mean')

            # Total loss
            total_loss = total_instance_loss + PRIOR_LOSS_WEIGHT * prior_loss

            # total_loss.backward()

            # Gradient Clipping
            # scaler.scale(total_loss).backward()
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), max_norm=1.0)
            # scaler.step(optimizer)
            # scaler.update()

            accelerator.backward(total_loss)
            if accelerator.sync_gradients:
                params_to_clip = (
                    itertools.chain(pipe.unet.parameters(), pipe.text_encoder.parameters())
                )
                accelerator.clip_grad_norm_(params_to_clip, max_norm=1.0)
            # accelerator.clip_grad_norm_(optimizer.parameters(), max_norm=1.0)
            optimizer.step()

            # Clear attention scores
            pipe.unet.attention_scores.clear()

            epoch_loss += total_loss.item()

        accelerator.wait_for_everyone()

        # Print losses
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}: Total Loss: {avg_epoch_loss:.4f}")
        torch.cuda.empty_cache()

    if accelerator.is_main_process:
        accelerator.wait_for_everyone()
        accelerator.unwrap_model(pipe.unet).save_pretrained(os.path.join(OUTPUT_DIR, "unet"))
        accelerator.unwrap_model(pipe.text_encoder).save_pretrained(os.path.join(OUTPUT_DIR, "text_encoder"))

    # Save trained weights
    # pipe.save_pretrained(OUTPUT_DIR)


# Main execution
if __name__ == "__main__":
    images, masks = load_data(IMAGES_DIR, MASKS_DIR)
    preprocessed_images, preprocessed_masks = preprocess_data(images, masks)
    instance_prompts = [INSTANCE_PROMPT] * len(preprocessed_images)

    # Create instance dataset and dataloader
    instance_dataset = DreamBoothDataset(preprocessed_images, preprocessed_masks, instance_prompts)
    train_dataloader = DataLoader(instance_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn,
                                  num_workers=2)

    # Prepare class images and create class dataset and dataloader
    class_images, class_prompts = prepare_class_images(pipe)
    preprocessed_class_images = preprocess_images(class_images)

    class_dataset = DreamBoothDataset(preprocessed_class_images, [None] * len(preprocessed_class_images), class_prompts)
    class_dataloader = DataLoader(class_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn,
                                  num_workers=2)

    # Training loop
    optimizer = AdamW(
        list(pipe.unet.parameters()) + list(pipe.text_encoder.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )
    noise_scheduler = DDPMScheduler()
    pipe.unet, pipe.text_encoder, optimizer, train_dataloader, class_dataloader = accelerator.prepare(
        pipe.unet,
        pipe.text_encoder,
        optimizer,
        train_dataloader,
        class_dataloader
    )

    # Run training
    print('TRAINING STARTED')
    train_loop(TRAIN_EPOCHS, noise_scheduler, optimizer, train_dataloader, class_dataloader)

    # Load the trained model
    # pipe = StableDiffusionPipeline.from_pretrained(OUTPUT_DIR)
    # pipe.to(DEVICE)
    #
    # # Generate and save an image
    # generated_image = pipe(INSTANCE_PROMPT).images[0]
    # generated_image.save(f"{OUTPUT_DIR}/final_output.png")
    # print("Training complete. Generated image saved.")
