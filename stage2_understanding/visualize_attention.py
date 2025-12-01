"""
👀 visualize_attention.py

Attention-map visualizer
Show model focusing on regions of an image

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
    attn_weights = []

    def get_attn(module, input, output):
        # PyTorch MultiheadAttention returns (attn_output, attn_output_weights) when need_weights=True
        # But in ViT, we need to hook into the attention mechanism differently
        # We'll modify the forward pass to capture attention
        attn_weights.append(output)

    # Hook into the last encoder block's self-attention
    last_block = model.encoder.layers[-1].self_attention

    # Temporarily modify the forward to return attention weights
    original_forward = last_block.forward

    def forward_with_attn(query, key, value, *args, **kwargs):
        # Call the original multi-head attention but request attention weights
        # Override need_weights to True to get attention weights
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        attn_output, attn_output_weights = original_forward(
            query, key, value, *args, **kwargs
        )
        attn_weights.append(attn_output_weights)
        return attn_output, attn_output_weights

    last_block.forward = forward_with_attn

    with torch.no_grad():
        _ = model(img_t)

    # Restore original forward
    last_block.forward = original_forward

    # attn shape: [batch, num_heads, num_tokens, num_tokens]
    attn = attn_weights[0]
    attn_map = attn.mean(1)[0, 0, 1:]  # average heads, take CLS -> patch attention

    # Reshape to grid (14x14 for ViT-B/16)
    side = int(attn_map.size(0) ** 0.5)
    attn_map = attn_map.reshape(side, side)
    attn_map = torch.nn.functional.interpolate(
        attn_map.unsqueeze(0).unsqueeze(0), size=(224, 224), mode="bilinear"
    )[0, 0]

    # Resize image to match attention map size
    img_resized = img.resize((224, 224), Image.BILINEAR)
    show_attention_on_image(img_resized, attn_map)


if __name__ == "__main__":
    main()
