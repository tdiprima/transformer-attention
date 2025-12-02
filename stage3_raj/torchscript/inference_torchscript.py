"""
Example script demonstrating how to load and use a TorchScript model for inference.
uv run inference_torchscript.py --model ../models_raj/vit_model.pt --image /path/to/image.png

uv run inference_torchscript.py \
    --model ../models_raj/vit_model.pt \
    --image /path/to/test/image.png \
    --class_names "Acinar tissue" "Dysplastic epithelium" "Fibrosis" "Lymph Aggregates" "Necrosis" "Nerves" "Normal ductal epithelium" "Reactive" "Stroma" "Tumor"
"""

import argparse

import torch
from PIL import Image
from rich_argparse import RichHelpFormatter
from torchvision import transforms


def load_torchscript_model(model_path, device):
    """
    Loads a TorchScript model from file.
    """
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


def preprocess_image(image_path, img_size=224):
    """
    Loads and preprocesses an image for ViT inference.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img)
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


def predict(model, image_tensor, device, class_names=None):
    """
    Runs inference on an image tensor.

    Args:
        model: TorchScript model
        image_tensor: preprocessed image tensor
        device: device to run on
        class_names: optional list of class names

    Returns:
        predicted class index and probabilities
    """
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor)

    # Get probabilities and prediction
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    pred_idx = probabilities.argmax().item()
    confidence = probabilities[pred_idx].item()

    return pred_idx, confidence, probabilities


def main():
    parser = argparse.ArgumentParser(
        description="TorchScript model inference",
        formatter_class=RichHelpFormatter,
        add_help=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="path to TorchScript model (.pt file)",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="path to image file",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="input image size (default 224)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--class_names",
        type=str,
        nargs="+",
        default=None,
        help="optional class names (e.g., --class_names classA classB classC)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="show top K predictions (default 5)",
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading TorchScript model from {args.model}...")
    model = load_torchscript_model(args.model, device)
    print("Model loaded successfully")

    # Preprocess image
    print(f"\nPreprocessing image: {args.image}")
    image_tensor = preprocess_image(args.image, args.img_size)
    print(f"Image tensor shape: {image_tensor.shape}")

    # Run inference
    print("\nRunning inference...")
    pred_idx, confidence, probabilities = predict(
        model, image_tensor, device, args.class_names
    )

    # Display results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)

    if args.class_names:
        pred_class = args.class_names[pred_idx]
        print(f"Predicted class: {pred_class} (index {pred_idx})")
    else:
        print(f"Predicted class index: {pred_idx}")

    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")

    # Show top-k predictions
    print(f"\nTop {args.top_k} predictions:")
    top_k_probs, top_k_indices = torch.topk(
        probabilities, min(args.top_k, len(probabilities))
    )

    for i, (prob, idx) in enumerate(zip(top_k_probs, top_k_indices), 1):
        idx_val = idx.item()
        prob_val = prob.item()
        if args.class_names and idx_val < len(args.class_names):
            class_name = args.class_names[idx_val]
            print(f"  {i}. {class_name:30s} {prob_val:.4f} ({prob_val*100:.2f}%)")
        else:
            print(f"  {i}. Class {idx_val:3d} {prob_val:.4f} ({prob_val*100:.2f}%)")

    print("=" * 60)


if __name__ == "__main__":
    main()
