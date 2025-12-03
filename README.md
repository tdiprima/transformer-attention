# Attention, Transformers! 🤖⚡️🔌💥

A hands-on learning lab where I work through **Vision Transformers** in PyTorch.  
The repo is split into stages so I can level up smoothly — from running a simple ViT,
to understanding how patch embeddings & ***attention*** work, and eventually building
a transformer-based classifier for Raj's dataset.

## 🚀 Learning Stages

### Stage 1 — Basics
Goal: Run a Vision Transformer on CIFAR-10 and understand the end-to-end workflow.

Contents:

- `train_vit_on_cifar10.py` — fine-tunes a pre-trained ViT-B/16
- Simple training loop, metrics, and model saving

### Stage 2 — Understanding Transformers
Goal: Demystify attention by implementing key pieces manually.

Planned contents:

- `patchify.py` — turn images into ViT-style patches
- `mini_vit.py` — tiny transformer encoder
- `visualize_attention.py` — visualize what the model attends to

### Stage 3 — Raj Classification Project
Goal: Build a Vision Transformer classifier for Raj's biomedical tile dataset.

Planned contents:

- Dataset loader for tile data
- Training + evaluation scripts
- Comparison against ResNet baseline

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🧪 Running the Stage 1 Training Script

```bash
python stage1_basics/train_vit_on_cifar10.py
```

This will:

* download CIFAR-10
* fine-tune a ViT
* save the model to `models/vit_cifar10.pth`

## 🔥 Why This Repo Exists

I'm leveling up into attention-based models so I can contribute to real transformer
work on biomedical images (Raj's classification data, spatial models, etc.).
This repo is my playground to learn, experiment, and build confidence before I jump into the deep end.

<br>
