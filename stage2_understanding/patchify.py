"""
🧩 patchify.py

Patchify images
Chops images into ViT-style patches

python patchify.py test_images/dog.jpg
"""

import sys

import torchvision.transforms as T
from PIL import Image


def patchify(img, patch_size=16):
    """
    img: Tensor [3, H, W]
    returns patches: [num_patches, patch_dim]
    """
    c, h, w = img.shape
    assert (
        h % patch_size == 0 and w % patch_size == 0
    ), "Image must be divisible by patch_size."

    patches = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    # shape: [3, h/ps, w/ps, ps, ps]

    patches = patches.permute(1, 2, 0, 3, 4).contiguous()
    # shape: [num_h, num_w, 3, ps, ps]

    patches = patches.view(-1, 3 * patch_size * patch_size)
    return patches


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python patchify.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

    img_tensor = transform(img)
    patches = patchify(img_tensor, patch_size=16)

    print("Image shape:", img_tensor.shape)  # Image shape: torch.Size([3, 224, 224])
    print("Patches shape:", patches.shape)  # Patches shape: torch.Size([196, 768])
    print("One patch vector length:", patches.shape[1])  # One patch vector length: 768
