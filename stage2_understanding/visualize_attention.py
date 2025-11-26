"""
👀 visualize_attention.py

This script:
loads a pretrained ViT-B/16
grabs the attention maps from the last attention block
overlays them on the input image so you can see what the model focused on

This is where transformers suddenly make sense.

python visualize_attention.py test_images/dog.jpg
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision import models


def show_attention_on_image(img, attn_map):
    attn_map = attn_map / attn_map.max()
    attn_map = attn_map.cpu().numpy()

    img_np = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    img_np = np.clip(img_np + attn_map[..., None] * 0.6, 0, 1)

    plt.figure(figsize=(6, 6))
    plt.imshow(img_np)
    plt.axis("off")
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_attention.py <image_path>")
        return

    image_path = sys.argv[1]
    img = Image.open(image_path).convert("RGB")

    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    img_t = transform(img).unsqueeze(0)

    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    model.eval()

    # Grab attention from the last block
    def get_attn(module, input, output):
        global attn
        attn = output

    last_block = model.encoder.layers[-1].self_attention.attention_probs
    handle = last_block.register_forward_hook(get_attn)

    with torch.no_grad():
        _ = model(img_t)

    handle.remove()

    # attn shape: [1, num_heads, num_tokens, num_tokens]
    attn_map = attn.mean(1)[0, 0, 1:]  # average heads, take CLS -> patch attention

    # Reshape to grid (14x14 for ViT-B/16)
    side = int(attn_map.size(0) ** 0.5)
    attn_map = attn_map.reshape(side, side)
    attn_map = torch.nn.functional.interpolate(
        attn_map.unsqueeze(0).unsqueeze(0), size=(224, 224), mode="bilinear"
    )[0, 0]

    show_attention_on_image(img, attn_map)


if __name__ == "__main__":
    main()
