"""
FaceShield and Adversarial Embedding Perturbation defenses.

This module implements proactive defenses against deepfake generation and
face swapping by disrupting facial feature extractors.
"""

from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import torch
import torchvision.transforms as T
import torchattacks
from torchvision.models import resnet50, ResNet50_Weights
from typing import Any
import cv2
from image_protection.utils import detect_faces

# Load a pre-trained model for feature extraction (e.g., ResNet-50)
# In a real scenario, a face recognition model like ArcFace would be used.
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()

# Define image transformations
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
])

def apply_faceshield(
    image: Image.Image,
    intensity: float = 0.5,
    method: str = "embedding_disruption",
    blur_radius: float = 1.0
) -> Image.Image:
    """
    Applies FaceShield-like protections to an image.
    This is a simplified simulation. A real implementation would be more complex.
    """
    image = image.convert("RGB") # Ensure image is in RGB format
    faces = detect_faces(image)
    
    if method == "attention_manipulation":
        # Simulate attention manipulation by adding noise to salient regions.
        img_array = np.array(image).astype(np.float32)
        h, w, _ = img_array.shape
        
        if faces:
            noise_mask = np.zeros((h, w, 1), dtype=np.float32)
            for (x, y, w_face, h_face) in faces:
                center_x, center_y = x + w_face // 2, y + h_face // 2
                radius = int(min(w_face, h_face) * 0.7 * intensity)
                cv2.circle(noise_mask, (center_x, center_y), radius, (1,), -1)
            
            noise = np.random.randn(h, w, 3) * intensity * 255 * noise_mask
            protected_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        else:
            protected_array = img_array.astype(np.uint8)
        
    elif method == "embedding_disruption":
        # Use PGD attack to perturb the image embeddings
        return apply_adversarial_embedding(image, attack_method="pgd", epsilon=intensity * 0.1)
        
    else:
        # Default to returning the original image if method is unknown
        return image

    # Apply Gaussian blur for imperceptibility
    if blur_radius > 0:
        protected_image = Image.fromarray(protected_array)
        if method == "attention_manipulation" and faces:
            # Blur only the face regions
            blurred_image = protected_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
            # Create a mask for compositing
            blur_mask_img = Image.new("L", image.size, 0)
            mask_draw = ImageDraw.Draw(blur_mask_img)

            for (x, y, w_face, h_face) in faces:
                # Draw filled rectangles for the face areas
                mask_draw.rectangle([x, y, x + w_face, y + h_face], fill=255)
            
            # Feather the mask to create a smooth transition
            blur_mask_img = blur_mask_img.filter(ImageFilter.GaussianBlur(radius=max(1, blur_radius * 2)))

            # Composite the blurred faces onto the protected image
            return Image.composite(blurred_image, protected_image, blur_mask_img)
        else:
            # For other methods or if no faces are found, blur the whole image
            return protected_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    return Image.fromarray(protected_array)


def apply_adversarial_embedding(
    image: Image.Image,
    attack_method: str = "pgd",
    epsilon: float = 0.03
) -> Image.Image:
    """
    Applies adversarial embedding perturbations using a specified attack.
    """
    # Normalize the image for the model
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_t = normalize(transform(image)).unsqueeze(0)
    
    # We need a target for the attack. Let's use a random target class.
    target = torch.LongTensor([np.random.randint(1000)])
    
    # Define the attack
    if attack_method == "pgd":
        # Projected Gradient Descent
        atk = torchattacks.PGD(model, eps=epsilon, alpha=epsilon/10, steps=40, random_start=True)
    else: # Default to FGSM
        # Fast Gradient Sign Method
        atk = torchattacks.FGSM(model, eps=epsilon)

    # Generate the adversarial perturbation
    perturbed_t = atk(img_t, target)
    
    # Convert back to PIL Image
    # The output of torchattacks is already in the [0,1] range, so we just need to convert to PIL
    pil_image = T.ToPILImage()(perturbed_t.squeeze(0).cpu())
    
    return pil_image


def apply_protection(
    image: Image.Image,
    **kwargs: Any
) -> Image.Image:
    """
    Main function to apply selected FaceShield/Adversarial protections.
    """
    intensity: float = kwargs.get("faceshield_intensity", 0.5)
    method: str = kwargs.get("faceshield_method", "attention_manipulation")
    blur_radius: float = kwargs.get("faceshield_blur", 1.0)
    
    return apply_faceshield(image, intensity, method, blur_radius)