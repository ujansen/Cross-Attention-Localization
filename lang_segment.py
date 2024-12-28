from PIL import Image
from lang_sam import LangSAM
import numpy as np

model = LangSAM()
image_pil = Image.open("./images/gandalf.jpeg").convert("RGB")

text_prompt = "man"

results = model.predict([image_pil], [text_prompt])

mask = results[0]['masks']

mask = np.squeeze(mask)
mask = (mask - mask.min()) / (mask.max() - mask.min()) * 255
mask = mask.astype(np.uint8)
np.save('./masks/gandalf.npy', mask)

segmentation_mask = Image.fromarray(mask)
# segmentation_mask.save('./masks/gandalf-smoking_mask.jpeg')
