# Transformers: A Comprehensive Guide

## What is a Transformer?

A **Transformer** is a neural network architecture introduced in the 2017 paper "Attention Is All You Need" by Vaswani et al. Originally designed for natural language processing (NLP), transformers have revolutionized machine learning and are now used across many domains including computer vision, speech recognition, and protein folding prediction.

The key innovation of transformers is the **self-attention mechanism**, which allows the model to weigh the importance of different parts of the input when processing each element, without relying on sequential processing like RNNs or local receptive fields like CNNs.

## Core Concepts

### 1. Self-Attention Mechanism

Self-attention allows each element in a sequence to "attend to" all other elements, determining which ones are most relevant for processing the current element.

**How it works:**

For each input element, we create three vectors:

- **Query (Q)**: What we're looking for
- **Key (K)**: What each element offers
- **Value (V)**: The actual content

The attention score is computed as:

```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
```

Where `d_k` is the dimension of the key vectors (used for scaling).

**Intuition:** Imagine reading a sentence and deciding which words are important for understanding a particular word. Self-attention does this automatically for the model.

### 2. Multi-Head Attention

Instead of performing attention once, we perform it multiple times in parallel with different learned projections. This allows the model to attend to information from different representation subspaces.

**Benefits:**

- Captures different types of relationships (e.g., syntactic, semantic)
- Increases model capacity
- Makes the model more robust

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W^O
where head_i = Attention(Q*W_i^Q, K*W_i^K, V*W_i^V)
```

### 3. Position Embeddings

Since transformers process all inputs simultaneously (unlike RNNs which process sequentially), they need explicit positional information. Positional embeddings are added to the input embeddings to give the model information about the order of elements.

**Types:**

- **Sinusoidal**: Fixed mathematical functions
- **Learned**: Trainable parameters (used in our implementation)

### 4. Feed-Forward Networks

After attention, each position passes through an identical feed-forward network:

```
FFN(x) = ReLU(x * W_1 + b_1) * W_2 + b_2
```

Modern transformers often use GELU activation instead of ReLU.

### 5. Layer Normalization and Residual Connections

Each sub-layer (attention and feed-forward) is wrapped with:

- **Residual connection**: Adds the input to the output to enable gradient flow
- **Layer normalization**: Normalizes activations for stable training

```
output = LayerNorm(x + Sublayer(x))
```

## Vision Transformers (ViT)

Vision Transformers adapt the transformer architecture for image classification. Instead of processing text tokens, they process image patches.

### How ViT Works

1. **Patch Embedding**
   - Split the image into fixed-size patches (e.g., 16x16 pixels)
   - Flatten each patch into a vector
   - Project to embedding dimension using a learned linear transformation

   ```
   For a 224x224 image with 16x16 patches:
   - Number of patches = (224/16)^2 = 196 patches
   - Each patch = 16*16*3 = 768 values (for RGB)
   ```

2. **Class Token**
   - Prepend a learnable [CLS] token to the sequence
   - This token aggregates information from all patches
   - Used for final classification

3. **Positional Embedding**
   - Add learned positional embeddings to retain spatial information
   - Without this, the model wouldn't know which patch came from where

4. **Transformer Encoder**
   - Process patches through multiple transformer blocks
   - Each block applies multi-head attention and feed-forward layers

5. **Classification Head**
   - Extract the [CLS] token from the final layer
   - Pass through a linear layer to get class predictions

## Architecture Diagram

```
Input Image (224x224x3)
        |
        v
  [Patch Embedding]
  Split into 196 patches (16x16 each)
  Project to embedding_dim
        |
        v
  [Prepend CLS token]
  [Add positional embeddings]
        |
        v
  [Transformer Block 1]
    - Multi-Head Attention
    - Add & Norm
    - Feed Forward
    - Add & Norm
        |
        v
  [Transformer Block 2]
        |
       ...
        |
        v
  [Transformer Block N]
        |
        v
  [Extract CLS token]
        |
        v
  [Classification Head]
  Linear(embed_dim → num_classes)
        |
        v
   Class Predictions
```

## Key Advantages of Transformers

1. **Parallelization**: Unlike RNNs, all positions can be processed simultaneously, making training much faster.

2. **Long-range Dependencies**: Self-attention can directly connect any two positions, regardless of distance.

3. **Interpretability**: Attention weights can be visualized to understand what the model focuses on.

4. **Flexibility**: The same architecture works across different domains (text, images, audio, etc.).

5. **Scalability**: Performance improves consistently with model size and data (scaling laws).

## Transformer vs CNN for Vision

| Aspect | CNN | Transformer |
|--------|-----|-------------|
| **Inductive Bias** | Strong (locality, translation invariance) | Weak (learns structure from data) |
| **Receptive Field** | Grows with depth | Global from first layer |
| **Data Efficiency** | Better with small datasets | Requires more data to train |
| **Computation** | Efficient for small images | Quadratic with sequence length |
| **Performance** | Excellent with proper data augmentation | State-of-the-art with large datasets |

## Implementation Details (from train_transformer.py)

Our implementation includes:

### PatchEmbedding
```python
# Uses Conv2d with kernel_size=stride=patch_size
# This efficiently splits and embeds patches in one operation
```

### MultiHeadAttention
```python
# Creates Q, K, V projections
# Computes scaled dot-product attention
# Applies softmax and combines with values
```

### TransformerBlock
```python
# Applies attention with residual connection
# Applies feed-forward with residual connection
# Layer normalization before each sub-layer (Pre-LN)
```

### VisionTransformer
```python
# Configurable depth, embed_dim, num_heads
# Learnable CLS token and positional embeddings
# Final classification from CLS token representation
```

## Hyperparameters Explained

- **embed_dim**: Dimension of token embeddings (e.g., 384, 768)
- **depth**: Number of transformer blocks (e.g., 6, 12)
- **num_heads**: Number of attention heads (e.g., 6, 12)
- **mlp_ratio**: Hidden layer expansion in feed-forward (typically 4.0)
- **patch_size**: Size of image patches (typically 16 or 32)
- **dropout**: Dropout rate for regularization

**Smaller models for limited data:**

- Use smaller embed_dim (384 instead of 768)
- Fewer layers (6 instead of 12)
- This reduces overfitting and speeds up training

## Training Tips

1. **Data Augmentation**: Crucial for ViT, especially with small datasets
   - Random crops, flips, rotations
   - Color jittering, cutout, mixup

2. **Optimizer**: AdamW with weight decay (0.05) works well

3. **Learning Rate**: Start with 1e-4, use warmup and cosine decay for longer training

4. **Batch Size**: Larger batches (64+) generally help, but adjust based on GPU memory

5. **Regularization**: Dropout, stochastic depth, label smoothing

## Applications Beyond Classification

Transformers are used in:

- Object Detection (DETR, Deformable DETR)
- Segmentation (SegFormer, Mask2Former)
- Image Generation (DALL-E, Stable Diffusion)
- Multi-modal Learning (CLIP, BLIP)
- Medical Imaging (our pathology classifier!)

## Further Reading

- Original Paper: "Attention Is All You Need" (Vaswani et al., 2017)
- ViT Paper: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
- The Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
- Hugging Face Transformers Documentation

## Mathematical Notation Reference

- **d_model**: Embedding dimension
- **d_k, d_v**: Dimensions of key and value vectors
- **h**: Number of attention heads
- **N**: Number of transformer blocks
- **n**: Sequence length (number of patches + 1 for ViT)
- **softmax**: `softmax(x_i) = exp(x_i) / sum(exp(x_j))`

---

This implementation provides a foundation for understanding and experimenting with transformer-based models for pathology image classification. The modular design allows easy experimentation with different configurations and extensions.
