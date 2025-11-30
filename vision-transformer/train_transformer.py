"""
Vision Transformer (ViT) for Image Classification
Self-contained script with transformer-based classifier
"""
import os
import random
from glob import glob
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ============================================================================
# Dataset
# ============================================================================
class ImgDataset(Dataset):
    def __init__(self, root_dir, classes, img_size=(224, 224), augment=True):
        self.samples = []
        for cls in classes:
            files = glob(os.path.join(root_dir, cls, "*"))
            for f in files:
                self.samples.append((f, classes.index(cls)))
        self.img_size = img_size
        self.augment = augment
        self.transform_train = T.Compose([
            T.Resize(img_size),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transform_val = T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, label = self.samples[idx]
        img = Image.open(p).convert("RGB")
        img = self.transform_train(img) if self.augment else self.transform_val(img)
        return img, label


# ============================================================================
# Transformer Components
# ============================================================================
class PatchEmbedding(nn.Module):
    """Split image into patches and embed them"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Convolutional layer to create patch embeddings
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (batch_size, channels, height, width)
        x = self.projection(x)  # (batch_size, embed_dim, n_patches**0.5, n_patches**0.5)
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.shape

        # Generate Q, K, V matrices
        qkv = self.qkv(x)  # (batch_size, seq_length, embed_dim * 3)
        qkv = qkv.reshape(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_length, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Apply attention to values
        out = torch.matmul(attention, v)  # (batch_size, num_heads, seq_length, head_dim)
        out = out.transpose(1, 2)  # (batch_size, seq_length, num_heads, head_dim)
        out = out.reshape(batch_size, seq_length, embed_dim)

        # Final projection
        out = self.projection(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    def __init__(self, embed_dim=768, hidden_dim=3072, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer encoder block"""
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, int(embed_dim * mlp_ratio), dropout)

    def forward(self, x):
        # Multi-head attention with residual connection
        x = x + self.attn(self.norm1(x))
        # Feed-forward with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
# Vision Transformer
# ============================================================================
class VisionTransformer(nn.Module):
    """Vision Transformer for image classification"""
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=10,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches

        # Class token (learnable)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embeddings (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        batch_size = x.shape[0]

        # Create patch embeddings
        x = self.patch_embed(x)  # (batch_size, n_patches, embed_dim)

        # Prepend class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, n_patches + 1, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Extract class token and classify
        x = self.norm(x)
        cls_token_final = x[:, 0]  # (batch_size, embed_dim)
        x = self.head(cls_token_final)

        return x


# ============================================================================
# Training Functions
# ============================================================================
def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total = 0
    for imgs, lbls in tqdm(loader, desc="Training"):
        imgs, lbls = imgs.to(device), lbls.to(device)
        preds = model(imgs)
        loss = loss_fn(preds, lbls)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
    return total / len(loader)


def validate(model, loader, loss_fn, device):
    model.eval()
    total = 0
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Validating"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, lbls)
            total += loss.item()
            correct += (preds.argmax(1) == lbls).sum().item()
            total_samples += imgs.size(0)
    return total / len(loader), correct / total_samples


# ============================================================================
# Main Training Loop
# ============================================================================
def main():
    # Configuration
    ROOT = "data/classification"
    CLASSES = sorted([d.name for d in Path(ROOT).iterdir() if d.is_dir()])
    IMG_SIZE = 224
    PATCH_SIZE = 16
    BATCH = 32
    EPOCHS = 25
    LR = 1e-4

    # Model configuration (smaller ViT for faster training)
    EMBED_DIM = 384
    DEPTH = 6
    NUM_HEADS = 6
    MLP_RATIO = 4.0
    DROPOUT = 0.1

    print(f"Found {len(CLASSES)} classes: {CLASSES}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"Number of patches: {(IMG_SIZE // PATCH_SIZE) ** 2}")

    # Build dataset
    dataset = ImgDataset(ROOT, CLASSES, img_size=(IMG_SIZE, IMG_SIZE), augment=True)
    random.shuffle(dataset.samples)
    split = int(0.8 * len(dataset))
    train_ds = torch.utils.data.Subset(dataset, range(0, split))
    val_ds = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    train_loader = DataLoader(
        train_ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True
    )

    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = VisionTransformer(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=3,
        num_classes=len(CLASSES),
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)

    # Training loop
    best_acc = 0
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        tr_loss = train_epoch(model, train_loader, opt, loss_fn, device)
        val_loss, acc = validate(model, val_loader, loss_fn, device)
        print(f"train_loss: {tr_loss:.4f} | val_loss: {val_loss:.4f} | val_acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_transformer.pth")
            print(f"Saved best_transformer.pth (acc: {best_acc:.4f})")

    print(f"\nTraining complete! Best validation accuracy: {best_acc:.4f}")

    # Export to ONNX
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    model.load_state_dict(torch.load("best_transformer.pth"))
    model.eval()
    torch.onnx.export(model, dummy, "transformer.onnx", opset_version=11)
    print("Exported transformer.onnx")


if __name__ == "__main__":
    main()
