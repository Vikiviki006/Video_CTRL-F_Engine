import torch
from transformers import AutoProcessor, SiglipModel

MODEL_NAME = "google/siglip2-so400m-patch14-384"

# Select GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load processor
processor = AutoProcessor.from_pretrained(MODEL_NAME)

# Load model
model = SiglipModel.from_pretrained(MODEL_NAME)

# Move model to GPU/CPU
model = model.to(device)

print("MODEL LOADED:", type(model))
print("DEVICE:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

from PIL import Image

image = Image.open("test.jpg").convert("RGB")

inputs = processor(
    images=image,
    return_tensors="pt"
)

inputs = {
    k: v.to(device) if hasattr(v, "to") else v
    for k, v in inputs.items()
}

with torch.no_grad():
    image_output = model.get_image_features(**inputs)

    if hasattr(image_output, "pooler_output"):
        image_features = image_output.pooler_output
    elif hasattr(image_output, "last_hidden_state"):
        image_features = image_output.last_hidden_state
    else:
        image_features = image_output

print("Image embedding shape:", image_features.shape)
print(type(image_output))
print(image_output)


with torch.no_grad():
    output = model.get_image_features(**inputs)

print("Type:", type(output))
print("Pooler:", output.pooler_output.shape if output.pooler_output is not None else None)
print("Last hidden:", output.last_hidden_state.shape if output.last_hidden_state is not None else None)


text = "a hot air ballon"

text_inputs = processor(
    text=[text],
    return_tensors="pt",
    padding="max_length"
)

text_inputs = {
    k: v.to(device) if hasattr(v, "to") else v
    for k, v in text_inputs.items()
}

with torch.no_grad():
    text_output = model.get_text_features(**text_inputs)

text_embedding = text_output.pooler_output

text_embedding = torch.nn.functional.normalize(
    text_embedding,
    p=2,
    dim=-1
)

print("Text embedding:", text_embedding.shape)
