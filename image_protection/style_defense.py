"""
Style Defense module for protecting images against style transfer and AI analysis.

This module implements various texture warping and blending techniques to
disrupt AI's ability to extract and transfer artistic styles.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
from typing import Tuple, Optional
import random


def apply_style_defense(
    image: Image.Image,
    strength: float = 0.5,
    texture_type: str = "noise"
) -> Image.Image:
    """
    Apply style defense techniques to protect against style transfer.
    
    Args:
        image: PIL Image to protect
        strength: Defense strength (0.0 to 1.0)
        texture_type: Type of texture defense ("noise", "blur", "pixelate", "swirl")
    
    Returns:
        Protected PIL Image
    """
    # Convert to numpy array for processing
    img_array = np.array(image)
    
    # Apply selected texture defense
    if texture_type == "noise":
        protected = _apply_noise_texture(img_array, strength)
    elif texture_type == "blur":
        protected = _apply_selective_blur(img_array, strength)
    elif texture_type == "pixelate":
        protected = _apply_adaptive_pixelation(img_array, strength)
    elif texture_type == "swirl":
        protected = _apply_swirl_distortion(img_array, strength)
    else:
        raise ValueError(f"Unknown texture type: {texture_type}")
    
    # Apply additional style disruption
    protected = _apply_style_mixing(protected, strength)
    
    # Convert back to PIL Image
    return Image.fromarray(protected.astype(np.uint8))


def _apply_noise_texture(
    img_array: np.ndarray,
    strength: float
) -> np.ndarray:
    """Apply sophisticated noise patterns to disrupt style extraction."""
    height, width = img_array.shape[:2]
    result = img_array.copy().astype(np.float32)
    
    # Generate multi-scale noise
    noise_scales = [1, 2, 4, 8]
    combined_noise = np.zeros_like(img_array, dtype=np.float32)
    
    for scale in noise_scales:
        # Generate noise at different scales
        noise = np.random.randn(height // scale, width // scale, 3)
        
        # Upscale noise
        noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_LINEAR)
        
        # Weight by scale (higher frequency = lower weight)
        weight = 1.0 / scale
        combined_noise += noise * weight
    
    # Normalize noise
    combined_noise = combined_noise / len(noise_scales)
    
    # Apply Perlin-like noise pattern
    perlin_noise = _generate_perlin_noise(height, width)
    if len(img_array.shape) == 3:
        perlin_noise = np.stack([perlin_noise] * 3, axis=-1)
    
    # Combine noises
    final_noise = combined_noise * 0.7 + perlin_noise * 0.3
    
    # Apply noise with edge preservation
    edges = _detect_edges(img_array)
    edge_mask = 1 - edges  # Less noise on edges
    
    if len(img_array.shape) == 3:
        edge_mask = np.stack([edge_mask] * 3, axis=-1)
    
    result += final_noise * strength * 30 * edge_mask
    
    return np.clip(result, 0, 255)


def _apply_selective_blur(
    img_array: np.ndarray,
    strength: float
) -> np.ndarray:
    """Apply selective blurring to disrupt fine style details."""
    result = img_array.copy()
    
    # Detect different image regions
    edges = _detect_edges(img_array)
    texture_map = _detect_texture_regions(img_array)
    
    # Create blur masks for different regions
    smooth_regions = 1 - np.maximum(edges, texture_map)
    
    # Apply different blur kernels
    blur_sizes = [
        (3, 3),
        (5, 5),
        (7, 7),
        (9, 9)
    ]
    
    # Progressive blurring based on region type
    for i, blur_size in enumerate(blur_sizes):
        blur_strength = strength * (i + 1) / len(blur_sizes)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(result, blur_size, 0)
        
        # Blend based on region mask
        mask = smooth_regions * blur_strength
        if len(img_array.shape) == 3:
            mask = np.stack([mask] * 3, axis=-1)
        
        result = result * (1 - mask) + blurred * mask
    
    # Apply bilateral filter for edge-preserving smoothing
    if strength > 0.5:
        result = cv2.bilateralFilter(
            result.astype(np.uint8),
            d=9,
            sigmaColor=75 * strength,
            sigmaSpace=75 * strength
        )
    
    return result.astype(np.uint8)


def _apply_adaptive_pixelation(
    img_array: np.ndarray,
    strength: float
) -> np.ndarray:
    """Apply adaptive pixelation that varies based on image content."""
    height, width = img_array.shape[:2]
    result = img_array.copy()
    
    # Detect image complexity
    complexity_map = _calculate_complexity_map(img_array)
    
    # Define pixelation levels
    min_block_size = max(2, int(4 * (1 - strength)))
    max_block_size = max(4, int(16 * strength))
    
    # Apply adaptive pixelation
    for y in range(0, height, min_block_size):
        for x in range(0, width, min_block_size):
            # Get local complexity
            local_complexity = np.mean(
                complexity_map[y:y+min_block_size, x:x+min_block_size]
            )
            
            # Determine block size based on complexity
            block_size = int(min_block_size + 
                           (max_block_size - min_block_size) * (1 - local_complexity))
            
            # Apply pixelation to block
            y_end = min(y + block_size, height)
            x_end = min(x + block_size, width)
            
            block = img_array[y:y_end, x:x_end]
            if block.size > 0:
                avg_color = np.mean(block, axis=(0, 1))
                result[y:y_end, x:x_end] = avg_color
    
    # Blend with original based on strength
    result = img_array * (1 - strength * 0.8) + result * (strength * 0.8)
    
    return result.astype(np.uint8)


def _apply_swirl_distortion(
    img_array: np.ndarray,
    strength: float
) -> np.ndarray:
    """Apply swirl distortion to disrupt spatial style patterns."""
    height, width = img_array.shape[:2]
    result = np.zeros_like(img_array)
    
    # Create multiple swirl centers
    num_swirls = max(1, int(5 * strength))
    swirl_centers = []
    
    for _ in range(num_swirls):
        cx = random.randint(width // 4, 3 * width // 4)
        cy = random.randint(height // 4, 3 * height // 4)
        swirl_centers.append((cx, cy))
    
    # Create coordinate grids
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Apply swirl transformation
    for cx, cy in swirl_centers:
        # Calculate distance from center
        dx = x - cx
        dy = y - cy
        distance = np.sqrt(dx**2 + dy**2)
        
        # Calculate swirl angle based on distance
        max_radius = min(width, height) / 3
        angle = strength * 2 * np.pi * (1 - distance / max_radius)
        angle[distance > max_radius] = 0
        
        # Apply rotation
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        new_x = cx + dx * cos_angle - dy * sin_angle
        new_y = cy + dx * sin_angle + dy * cos_angle
        
        # Update coordinates
        x = new_x
        y = new_y
    
    # Clip coordinates
    x = np.clip(x, 0, width - 1).astype(np.int32)
    y = np.clip(y, 0, height - 1).astype(np.int32)
    
    # Remap pixels
    result = img_array[y, x]
    
    # Smooth transitions
    result = cv2.GaussianBlur(result, (3, 3), 0)
    
    return result


def _apply_style_mixing(
    img_array: np.ndarray,
    strength: float
) -> np.ndarray:
    """Mix different style disruption techniques."""
    result = img_array.copy().astype(np.float32)
    
    # Color channel shuffling
    if strength > 0.3 and len(img_array.shape) == 3:
        # Randomly adjust channel weights
        channel_weights = 1 + (np.random.rand(3) - 0.5) * strength * 0.3
        for i in range(3):
            result[:, :, i] *= channel_weights[i]
    
    # Local contrast manipulation
    result = _manipulate_local_contrast(result, strength)
    
    # Texture synthesis disruption
    texture_noise = _generate_texture_noise(result.shape[:2])
    if len(img_array.shape) == 3:
        texture_noise = np.stack([texture_noise] * 3, axis=-1)
    
    result += texture_noise * strength * 20
    
    return np.clip(result, 0, 255).astype(np.uint8)


def _generate_perlin_noise(
    height: int,
    width: int,
    scale: float = 100.0
) -> np.ndarray:
    """Generate Perlin-like noise pattern."""
    # Simplified Perlin noise using multiple octaves
    noise = np.zeros((height, width))
    
    octaves = 4
    persistence = 0.5
    
    for octave in range(octaves):
        freq = 2 ** octave
        amp = persistence ** octave
        
        # Generate random gradients
        grid_size = max(2, int(scale / freq))
        grid_h = height // grid_size + 2
        grid_w = width // grid_size + 2
        
        gradients = np.random.randn(grid_h, grid_w, 2)
        gradients /= np.linalg.norm(gradients, axis=2, keepdims=True)
        
        # Interpolate
        octave_noise = np.zeros((height, width))
        
        for y in range(height):
            for x in range(width):
                # Grid coordinates
                gx = x / grid_size
                gy = y / grid_size
                
                # Grid cell coordinates
                x0, y0 = int(gx), int(gy)
                x1, y1 = x0 + 1, y0 + 1
                
                # Interpolation weights
                wx = gx - x0
                wy = gy - y0
                
                # Smooth interpolation
                wx = wx * wx * (3 - 2 * wx)
                wy = wy * wy * (3 - 2 * wy)
                
                # Compute dot products
                if x1 < grid_w and y1 < grid_h:
                    n00 = gradients[y0, x0, 0] * (gx - x0) + gradients[y0, x0, 1] * (gy - y0)
                    n10 = gradients[y0, x1, 0] * (gx - x1) + gradients[y0, x1, 1] * (gy - y0)
                    n01 = gradients[y1, x0, 0] * (gx - x0) + gradients[y1, x0, 1] * (gy - y1)
                    n11 = gradients[y1, x1, 0] * (gx - x1) + gradients[y1, x1, 1] * (gy - y1)
                    
                    # Interpolate
                    nx0 = n00 * (1 - wx) + n10 * wx
                    nx1 = n01 * (1 - wx) + n11 * wx
                    octave_noise[y, x] = nx0 * (1 - wy) + nx1 * wy
        
        noise += octave_noise * amp
    
    # Normalize to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    
    return noise


def _detect_edges(img_array: np.ndarray) -> np.ndarray:
    """Detect edges in the image."""
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    
    # Normalize and blur
    edges = edges.astype(np.float32) / 255.0
    edges = cv2.GaussianBlur(edges, (5, 5), 0)
    
    return edges


def _detect_texture_regions(img_array: np.ndarray) -> np.ndarray:
    """Detect textured regions in the image."""
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Calculate local standard deviation
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    
    mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    sq_mean = cv2.filter2D(gray.astype(np.float32) ** 2, -1, kernel)
    variance = sq_mean - mean ** 2
    std_dev = np.sqrt(np.maximum(variance, 0))
    
    # Normalize
    texture_map = std_dev / (std_dev.max() + 1e-8)
    
    return texture_map


def _calculate_complexity_map(img_array: np.ndarray) -> np.ndarray:
    """Calculate local complexity of image regions."""
    edges = _detect_edges(img_array)
    texture = _detect_texture_regions(img_array)
    
    # Combine edge and texture information
    complexity = edges * 0.5 + texture * 0.5
    
    # Smooth the complexity map
    complexity = cv2.GaussianBlur(complexity, (9, 9), 0)
    
    return complexity


def _manipulate_local_contrast(
    img_array: np.ndarray,
    strength: float
) -> np.ndarray:
    """Manipulate local contrast to disrupt style patterns."""
    result = img_array.copy()
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    if len(img_array.shape) == 3:
        # Convert to LAB color space
        lab = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(
            clipLimit=2.0 + strength * 3,
            tileGridSize=(8, 8)
        )
        l = clahe.apply(l)
        
        # Merge and convert back
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(
            clipLimit=2.0 + strength * 3,
            tileGridSize=(8, 8)
        )
        result = clahe.apply(img_array.astype(np.uint8))
    
    # Blend with original
    result = img_array * (1 - strength * 0.5) + result * (strength * 0.5)
    
    return result


def _generate_texture_noise(shape: Tuple[int, int]) -> np.ndarray:
    """Generate texture-based noise pattern."""
    height, width = shape
    
    # Create base texture pattern
    texture = np.zeros((height, width))
    
    # Add multiple frequency components
    frequencies = [(3, 3), (7, 5), (11, 13)]
    
    for fx, fy in frequencies:
        x = np.linspace(0, fx * np.pi, width)
        y = np.linspace(0, fy * np.pi, height)
        X, Y = np.meshgrid(x, y)
        
        pattern = np.sin(X) * np.cos(Y) + np.sin(X + Y)
        texture += pattern / len(frequencies)
    
    # Add random variations
    texture += np.random.randn(height, width) * 0.1
    
    # Normalize
    texture = (texture - texture.mean()) / (texture.std() + 1e-8)
    
    return texture