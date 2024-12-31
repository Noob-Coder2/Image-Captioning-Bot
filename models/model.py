import torch
from transformers import CLIPModel, CLIPProcessor, GPT2LMHeadModel, GPT2Tokenizer
from models.clip_gpt_bridge import CLIP2GPT

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Load GPT2 model
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load CLIP2GPT bridge
clip_to_gpt = CLIP2GPT(clip_dim=512, gpt_dim=gpt_model.config.n_embd).to(device)

def generate_caption(image_embedding):
    gpt_input = clip_to_gpt(image_embedding)
    input_ids = tokenizer.encode("<SOS>", return_tensors="pt").to(device)
    outputs = gpt_model.generate(input_ids=input_ids, max_length=50, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
