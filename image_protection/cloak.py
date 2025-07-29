"""
Cloaking module for anti-AI protection.

This module implements cloaking techniques inspired by Fawkes and LowKey
to add imperceptible perturbations that prevent AI from learning from images.
"""

import numpy as np
import numpy.typing as npt
from PIL import Image
import cv2
from image_protection.cropper import _detect_face_center


def apply_cloaking(
    image: Image.Image,
    intensity: float = 0.5,
    method: str = "fawkes"
) -> Image.Image:
    """
    Apply cloaking perturbations to protect against AI recognition.
    
    Args:
        image: PIL Image to protect
        intensity: Cloaking intensity (0.0 to 1.0)
        method: Cloaking method ("fawkes" or "lowkey")
    
    Returns:
        Protected PIL Image
    """
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    if method == "fawkes":
        protected_array = _apply_fawkes_style_cloaking(img_array, intensity)
    elif method == "lowkey":
        protected_array = _apply_lowkey_style_cloaking(img_array, intensity)
    else:
        raise ValueError(f"Unknown cloaking method: {method}")
    
    # Convert back to PIL Image
    return Image.fromarray(protected_array.astype(np.uint8))


def _apply_fawkes_style_cloaking(
    img_array: npt.NDArray[np.uint8],
    intensity: float
) -> npt.NDArray[np.uint8]:
    """
    Apply Fawkes-inspired cloaking (simplified version).
    
    Note: This is a simplified implementation. The real Fawkes uses
    adversarial perturbations computed against specific face recognition models.
    """
    height, width = img_array.shape[:2]
    
    # Generate structured noise pattern
    noise = _generate_adversarial_pattern(height, width, intensity)
    
    # Apply noise primarily to face-like regions (simplified detection)
    if len(img_array.shape) == 3:
        # For color images
        face_mask = np.zeros(img_array.shape[:2], dtype=np.float32)
        face_center = _detect_face_center(Image.fromarray(img_array))
        if face_center:
            cx, cy = face_center
            h, w = img_array.shape[:2]
            y, x = np.ogrid[:h, :w]
            face_mask = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * (min(h, w) / 4)**2))
        
        # Expand mask to 3 channels
        face_mask_3d = np.stack([face_mask] * 3, axis=-1)
        
        # Apply perturbation with face-aware masking
        perturbed = img_array.astype(np.float32)
        perturbed += noise * face_mask_3d * intensity * 15
    else:
        # For grayscale images
        perturbed = img_array.astype(np.float32)
        perturbed += noise[:, :, 0] * intensity * 15
    
    # Clip values to valid range
    perturbed = np.clip(perturbed, 0, 255)
    
    return perturbed.astype(np.uint8)


def _apply_lowkey_style_cloaking(
    img_array: npt.NDArray[np.uint8],
    intensity: float
) -> npt.NDArray[np.uint8]:
    """
    Apply LowKey-inspired cloaking (simplified version).
    
    LowKey focuses on privacy-preserving transformations that maintain
    visual quality while disrupting AI recognition.
    """
    # Apply frequency domain perturbations
    perturbed = _frequency_domain_perturbation(img_array, intensity).astype(np.float32)
    
    # Add edge-aware noise
    edge_noise = _generate_edge_aware_noise(img_array, intensity)
    perturbed += edge_noise
    
    # Clip values
    perturbed = np.clip(perturbed, 0, 255)
    
    return perturbed.astype(np.uint8)


def _generate_adversarial_pattern(
    height: int,
    width: int,
    intensity: float
) -> npt.NDArray[np.float32]:
    """Generate an adversarial noise pattern."""
    # Create base noise
    noise = np.random.randn(height, width, 3)
    
    # Apply Gaussian blur for smoothness
    noise = cv2.GaussianBlur(noise, (5, 5), 0)
    
    # Create structured patterns
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    
    # Add sinusoidal patterns (common in adversarial examples)
    pattern1 = np.sin(2 * np.pi * 3 * x) * np.cos(2 * np.pi * 3 * y)
    pattern2 = np.sin(2 * np.pi * 5 * x + np.pi/4) * np.cos(2 * np.pi * 5 * y + np.pi/4)
    
    # Combine patterns
    structured_noise = np.stack([
        noise[:, :, 0] + pattern1 * 0.3,
        noise[:, :, 1] + pattern2 * 0.3,
        noise[:, :, 2] + (pattern1 + pattern2) * 0.15
    ], axis=-1).astype(np.float32)
    
    # Normalize
    structured_noise = structured_noise * intensity
    
    return structured_noise


def _frequency_domain_perturbation(
    img_array: npt.NDArray[np.uint8],
    intensity: float
) -> npt.NDArray[np.float32]:
    """Apply perturbations in frequency domain."""
    result = img_array.copy().astype(np.float32)
    
    # Process each channel
    channels = 1 if len(img_array.shape) == 2 else img_array.shape[2]
    
    for c in range(channels):
        if channels == 1:
            channel = img_array
        else:
            channel = img_array[:, :, c]
        
        # Apply FFT
        f_transform = np.fft.fft2(channel)
        f_shift = np.fft.fftshift(f_transform)
        
        # Create frequency mask
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2
        
        # Modify high frequencies (details that AI might use)
        mask = np.ones((rows, cols), dtype=np.float32)
        r = min(rows, cols) // 4
        
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - ccol)**2 + (y - crow)**2 > r**2
        mask[mask_area] = 1 + np.random.uniform(-intensity, intensity, size=int(np.sum(mask_area)))
        
        # Apply mask
        f_shift = f_shift * mask
        
        # Inverse FFT
        f_ishift = np.fft.ifftshift(f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.real(img_back)
        
        if channels == 1:
            result = img_back
        else:
            result[:, :, c] = img_back
    
    return result.astype(np.float32)


def _generate_edge_aware_noise(
    img_array: npt.NDArray[np.uint8],
    intensity: float
) -> npt.NDArray[np.float32]:
    """Generate noise that's stronger near edges."""
    # Convert to grayscale for edge detection
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Detect edges
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    edges = edges.astype(np.float32) / 255.0
    
    # Dilate edges
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    # Blur for smooth transition
    edges = cv2.GaussianBlur(edges, (15, 15), 0)
    
    # Generate noise
    noise = np.random.randn(*img_array.shape) * intensity * 10
    
    # Apply edge mask
    if len(img_array.shape) == 3:
        edge_mask = np.stack([edges] * 3, axis=-1)
    else:
        edge_mask = edges
    
    edge_aware_noise = noise * edge_mask
    
    return edge_aware_noise


def validate_cloaking_effectiveness(
    original: Image.Image,
    cloaked: Image.Image
) -> dict:
    """
    Validate the effectiveness of cloaking.
    
    Returns metrics about the cloaking quality.
    """
    orig_array = np.array(original)
    cloak_array = np.array(cloaked)
    
    # Calculate PSNR (Peak Signal-to-Noise Ratio)
    mse = np.mean((orig_array - cloak_array) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # Calculate SSIM (Structural Similarity Index)
    # Simplified version
    ssim = _calculate_ssim(orig_array, cloak_array)
    
    # Calculate average perturbation
    avg_perturbation = np.mean(np.abs(orig_array.astype(np.float32) - cloak_array.astype(np.float32)))
    
    return {
        "psnr": psnr,
        "ssim": ssim,
        "avg_perturbation": avg_perturbation,
        "quality_assessment": _assess_quality(psnr, ssim)
    }


def _calculate_ssim(img1: npt.NDArray[np.uint8], img2: npt.NDArray[np.uint8]) -> float:
    """Simplified SSIM calculation."""
    # Constants
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    img1_64 = img1.astype(np.float64)
    img2_64 = img2.astype(np.float64)
    
    # Calculate means
    mu1 = cv2.GaussianBlur(img1_64, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2_64, (11, 11), 1.5)
    
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    # Calculate variances
    sigma1_sq = cv2.GaussianBlur(img1_64**2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2_64**2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1_64 * img2_64, (11, 11), 1.5) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    
    return np.mean(ssim_map)


def _assess_quality(psnr: float, ssim: float) -> str:
    """Assess the quality of cloaking based on metrics."""
    if psnr > 40 and ssim > 0.95:
        return "Excellent - Imperceptible protection"
    elif psnr > 35 and ssim > 0.90:
        return "Good - Minimal visual impact"
    elif psnr > 30 and ssim > 0.85:
        return "Fair - Slight visual changes"
    else:
        return "Strong - Visible protection applied"