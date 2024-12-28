import argparse
import itertools
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

import PIL
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from pathlib import Path
from torchvision import transforms

from accelerate import Accelerator

import bitsandbytes as bnb

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
instance = "mug"
new_token = "[V]"

instance_prompt = f"a {new_token} {instance}"
instance_data_directory = f"./{instance}_images/"

OUTPUT_DIR = f"./dreambooth_{instance}/"

images = []
for filename in os.listdir(instance_data_directory):
    filepath = os.path.join(instance_data_directory, filename)
    im = Image.open(filepath)
    images.append(im)

num_class_images = 80
sample_batch_size = 1
prior_loss_weight = 1

class_prompt = f"a photo of a {instance}"
prior_preservation_class_folder = f"./{instance}"
class_data_dir=prior_preservation_class_folder

learning_rate = 1e-6
batch_size = 1
max_train_steps=600

# Load Stable Diffusion pre-trained Model
pretrained_model_name = "CompVis/stable-diffusion-v1-4"
text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name, subfolder="text_encoder"
)
vae = AutoencoderKL.from_pretrained(
    pretrained_model_name, subfolder="vae"
)
unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name, subfolder="unet"
)
tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_name,
    subfolder="tokenizer",
)

num_added_tokens = tokenizer.add_tokens(new_token)
if num_added_tokens > 0:
    text_encoder.resize_token_embeddings(len(tokenizer))

# Initialize Accelerator
accelerator = Accelerator(
    mixed_precision="fp16",  # or "bf16" if supported
    gradient_accumulation_steps=3,  # Adjust based on your memory constraints
    cpu=False  # Set to True if you want to run on CPU
)

class DreamBoothDataset(Dataset):
    def __init__(
            self,
            instance_data_root,
            instance_prompt,
            tokenizer,
            class_data_root=None,
            class_prompt=None,
            size=512,
            center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(Path(class_data_root).iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = self.num_instance_images
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding='max_length',
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.class_data_root:
            idx = np.random.randint(0, len(self.class_images_path))
            class_image = Image.open(self.class_images_path[idx])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding='max_length',
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example


class PromptDataset(Dataset):
    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example

#Generate class images
def generate_class_images(class_data_root, num_class_images, pretrained_model_name, class_prompt):
    class_images_dir = Path(class_data_root)
    if not class_images_dir.exists():
        class_images_dir.mkdir(parents=True)
    cur_class_images = len(list(class_images_dir.iterdir()))

    if cur_class_images < num_class_images:
        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name, revision="fp16", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.enable_attention_slicing()
        pipeline.set_progress_bar_config(disable=True)

        num_new_images = num_class_images - cur_class_images
        print(f"Number of class images to sample: {num_new_images}.")

        sample_dataset = PromptDataset(class_prompt, num_new_images)
        sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=sample_batch_size)

        for example in tqdm(sample_dataloader, desc="Generating class images"):
            images = pipeline(example["prompt"]).images

            for i, image in enumerate(images):
                image.save(class_images_dir / f"{example['index'][i] + cur_class_images}.jpg")
        pipeline = None
        del pipeline
        with torch.no_grad():
          torch.cuda.empty_cache()

def collate_fn(examples):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    # concat class and instance examples for prior preservation
    input_ids += [example["class_prompt_ids"] for example in examples]
    pixel_values += [example["class_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = tokenizer.pad(
        {"input_ids": input_ids},
        padding="max_length",
        return_tensors="pt",
        max_length=tokenizer.model_max_length
    ).input_ids

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }
    return batch

generate_class_images(class_data_dir, num_class_images, pretrained_model_name, class_prompt)

vae.requires_grad_(False)
# text_encoder.requires_grad_(False)

optimizer_class = bnb.optim.AdamW8bit
params_to_optimize = itertools.chain(unet.parameters(), text_encoder.parameters())
optimizer = optimizer_class(
    params_to_optimize,
    lr=learning_rate,
    weight_decay=1e-4
)
noise_scheduler = DDPMScheduler.from_config(pretrained_model_name, subfolder="scheduler")

train_dataset = DreamBoothDataset(
    instance_data_root=instance_data_directory,
    instance_prompt=instance_prompt,
    class_data_root=class_data_dir,
    class_prompt=class_prompt,
    tokenizer=tokenizer,
    size=512,
    center_crop=True,
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)

lr_scheduler = get_scheduler(
    "constant",
    optimizer=optimizer,
    num_training_steps=max_train_steps
)

weight_dtype = torch.float16

vae.to(accelerator.device, dtype=weight_dtype)
vae.decoder.to("cpu")

unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    unet, text_encoder, optimizer, train_dataloader, lr_scheduler
)

print(len(train_dataset))

for epoch in tqdm(range(max_train_steps), desc='Training', file=sys.stdout):
    unet.train()
    text_encoder.train()
    epoch_loss = 0
    for step, batch in enumerate(train_dataloader):
        latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = text_encoder(batch["input_ids"])[0]

        # Predict the noise residual
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample

        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
        target, target_prior = torch.chunk(target, 2, dim=0)

        # Compute instance loss
        snr = compute_snr(noise_scheduler, timesteps)
        base_weight = (
                torch.stack([snr, 5.0 * torch.ones_like(snr)], dim=1).min(dim=1)[0] / snr
        )
        mse_loss_weight = base_weight
        denoise_loss = F.mse_loss(noise_pred.float(), target.float(), reduction='none')
        denoise_loss = denoise_loss.mean([1,2,3]) * mse_loss_weight
        loss = denoise_loss.mean()

        # Compute prior loss
        prior_loss = F.mse_loss(noise_pred_prior.float(), target_prior.float(), reduction="mean")

        # Add the prior loss to the instance loss.
        loss = loss + prior_loss_weight * prior_loss

        accelerator.backward(loss)

        if accelerator.sync_gradients:
            params_to_clip = (
                itertools.chain(unet.parameters(), text_encoder.parameters())
            )
            accelerator.clip_grad_norm_(params_to_clip, max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        epoch_loss += loss

    accelerator.wait_for_everyone()

    avg_epoch_loss = epoch_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}: Total Loss: {avg_epoch_loss:.4f}")
    torch.cuda.empty_cache()

if accelerator.is_main_process:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    accelerator.unwrap_model(unet).save_pretrained(os.path.join(OUTPUT_DIR, "unet"))
    accelerator.unwrap_model(text_encoder).save_pretrained(os.path.join(OUTPUT_DIR, "text_encoder"))