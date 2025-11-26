## Stage 2 — "Okay bet, now let's understand what's happening."

Goal: Peel the transformer open a bit without going full PhD.

Tasks:

1. **Write a tiny "manual vision transformer" module:**

   * Patchify an image manually
   * Add positional embeddings
   * Run it through a tiny `nn.TransformerEncoder`
   * Do classification

2. **Generate and visualize attention maps**

   * Grab attention from the model
   * Display which parts of the image it focuses on

3. **Swap between architectures**

   * ViT-base
   * ViT-small
   * Data-efficient ViT
   * Maybe DeiT (Distilled Vision Transformer)

You'll learn:

* Patch embeddings
* Positional embeddings
* Self-attention shapes
* Multi-head attention
* Why transformers get the whole-image context instantly

This is the "oooohhh THAT'S how they think" stage.

---

## Image credits:

### 🐶 1. Dog Close-Up (super clear subject)
dog.jpg https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg

### 🧍 2. Person Portrait (ViT LOVES faces)
person.jpg https://images.pexels.com/photos/1181686/pexels-photo-1181686.jpeg

### 🌸 3. Flower Macro (simple shapes + great contrast)
flower.jpg https://images.pexels.com/photos/462118/pexels-photo-462118.jpeg

### 🚗 4. Car (strong edges + recognizable ImageNet class)
car.jpg https://images.pexels.com/photos/358070/pexels-photo-358070.jpeg

### 🐻 5. Bear (because my name deserves attention maps too)
bear.jpg https://images.pexels.com/photos/145939/pexels-photo-145939.jpeg

<br>
