from PIL import Image
from transformers import CLIPProcessor, CLIPModel, ViTImageProcessor, ViTModel
import torch
import os

prompt = 'a mug in the desert'
generated_path = '/w/331/usen/Vanilla_DreamBooth/dreambooth_mug/mug_desert'
real_path = '/w/331/usen/Vanilla_DreamBooth/mug_images'



clip_model_name = 'openai/clip-vit-base-patch32'
vit_model_name = 'facebook/dino-vits16'
generated_images = [os.path.join(generated_path, image) for image in os.listdir(generated_path)]
real_images = [os.path.join(real_path, image) for image in os.listdir(real_path)]

clip_model = CLIPModel.from_pretrained(clip_model_name)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

dino_model = ViTModel.from_pretrained(vit_model_name)
dino_processor = ViTImageProcessor.from_pretrained(vit_model_name)

def calculate_embeddings_clip(images):
    embeddings = []
    for img in images:
        image = Image.open(img).convert('RGB')
        inputs = clip_processor(images=image, return_tensors='pt')
        with torch.no_grad():
            embedding = clip_model.get_image_features(**inputs)
            embeddings.append(embedding / embedding.norm(dim=-1, keepdim=True))
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings


def calculate_embeddings_dino(images):
    embeddings = []
    for img in images:
        image = Image.open(img).convert('RGB')
        inputs = dino_processor(images=image, return_tensors='pt')
        with torch.no_grad():
            embedding = dino_model(**inputs).last_hidden_state[:, 0, :]
            embeddings.append(embedding / embedding.norm(dim=-1, keepdim=True))
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings



real_embeddings_clip = calculate_embeddings_clip(real_images)
generated_embeddings_clip = calculate_embeddings_clip(generated_images)

cosine_similarity_clip = torch.mm(real_embeddings_clip, generated_embeddings_clip.T)

average_cosine_similarity_clip = cosine_similarity_clip.mean().item()
print(f'Cosine Similarity (CLIP): {average_cosine_similarity_clip}')

real_embeddings_dino = calculate_embeddings_dino(real_images)
generated_embeddings_dino = calculate_embeddings_dino(generated_images)

cosine_similarity_dino = torch.mm(real_embeddings_dino, generated_embeddings_dino.T)

average_cosine_similarity_dino = cosine_similarity_dino.mean().item()
print(f'Cosine Similarity (DINO): {average_cosine_similarity_dino}')

prompt_images = [os.path.join(generated_path, image) for image in os.listdir(generated_path)]

def calcuate_prompt_fidelity(images, prompt):
    inputs = clip_processor(text=prompt, return_tensors='pt', padding=True)
    with torch.no_grad():
        text_embeddings = clip_model.get_text_features(**inputs)

    text_embedding = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True).unsqueeze(0)
    image_embeddings = calculate_embeddings_clip(images)

    return (text_embedding @ image_embeddings.T).squeeze(0).mean().item()

prompt_fidelity = calcuate_prompt_fidelity(prompt_images, prompt)
print(f'Prompt Fidelity: {prompt_fidelity}')