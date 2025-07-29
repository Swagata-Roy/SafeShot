"""
Smart cropping module with edge softening capabilities.

This module provides intelligent cropping functions that can remove
identifying features while maintaining image aesthetics.
"""

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFilter
import cv2
from typing import Any
import mediapipe as mp  # type: ignore
from typing import Tuple, Optional
import math

# Initialize MediaPipe Face Detection
mp_face_detection: Any = mp.solutions.face_detection  # type: ignore
mp_drawing: Any = mp.solutions.drawing_utils  # type: ignore


def smart_crop(
    image: Image.Image,
    aspect_ratio: str = "Original",
    edge_softness: int = 10,
    focus_mode: str = "center"
) -> Image.Image:
    """
    Apply smart cropping with edge softening.
    
    Args:
        image: PIL Image to crop
        aspect_ratio: Target aspect ratio ("1:1", "4:3", "16:9", "9:16", "Original")
        edge_softness: Pixels for edge softening (0-50)
        focus_mode: Focus detection mode ("center", "face", "saliency")
    
    Returns:
        Cropped PIL Image with soft edges
    """
    # Get original dimensions
    width, height = image.size
    
    # Calculate target dimensions
    target_width, target_height = _calculate_target_dimensions(
        width, height, aspect_ratio
    )
    
    # Find optimal crop region
    crop_box = _find_optimal_crop(
        image, target_width, target_height, focus_mode
    )
    
    # Crop the image
    cropped = image.crop(crop_box)
    
    # Apply edge softening if requested
    if edge_softness > 0:
        cropped = _apply_edge_softening(cropped, edge_softness)
    
    return cropped


def privacy_crop(
    image: Image.Image,
    remove_edges: bool = True,
    blur_periphery: bool = True,
    crop_percentage: float = 0.1
) -> Image.Image:
    """
    Apply privacy-focused cropping to remove potentially identifying edges.
    
    Args:
        image: PIL Image to process
        remove_edges: Whether to crop out edges
        blur_periphery: Whether to blur the periphery
        crop_percentage: Percentage to crop from each edge (0.0-0.3)
    
    Returns:
        Privacy-cropped PIL Image
    """
    width, height = image.size
    
    if remove_edges:
        # Calculate crop amounts
        crop_x = int(width * crop_percentage)
        crop_y = int(height * crop_percentage)
        
        # Crop the image
        crop_box = (
            crop_x,
            crop_y,
            width - crop_x,
            height - crop_y
        )
        image = image.crop(crop_box)
    
    if blur_periphery:
        image = _blur_periphery(image)
    
    return image


def _calculate_target_dimensions(
    width: int,
    height: int,
    aspect_ratio: str
) -> Tuple[int, int]:
    """Calculate target dimensions based on aspect ratio."""
    if aspect_ratio == "Original":
        return width, height
    
    # Parse aspect ratio
    ratio_map = {
        "1:1": (1, 1),
        "4:3": (4, 3),
        "16:9": (16, 9),
        "9:16": (9, 16)
    }
    
    if aspect_ratio not in ratio_map:
        return width, height
    
    target_w_ratio, target_h_ratio = ratio_map[aspect_ratio]
    target_aspect = target_w_ratio / target_h_ratio
    current_aspect = width / height
    
    if current_aspect > target_aspect:
        # Image is wider than target - crop width
        target_width = int(height * target_aspect)
        target_height = height
    else:
        # Image is taller than target - crop height
        target_width = width
        target_height = int(width / target_aspect)
    
    return target_width, target_height


def _find_optimal_crop(
    image: Image.Image,
    target_width: int,
    target_height: int,
    focus_mode: str
) -> Tuple[int, int, int, int]:
    """Find the optimal crop region based on focus mode."""
    width, height = image.size
    
    if focus_mode == "center":
        # Center crop
        left = (width - target_width) // 2
        top = (height - target_height) // 2
    
    elif focus_mode == "face":
        # Try to detect faces and crop around them
        face_center = _detect_face_center(image)
        if face_center:
            cx, cy = face_center
            left = max(0, min(width - target_width, cx - target_width // 2))
            top = max(0, min(height - target_height, cy - target_height // 2))
        else:
            # Fallback to center crop
            left = (width - target_width) // 2
            top = (height - target_height) // 2
    
    elif focus_mode == "saliency":
        # Use saliency detection to find interesting regions
        salient_center = _detect_salient_region(image)
        cx, cy = salient_center
        left = max(0, min(width - target_width, cx - target_width // 2))
        top = max(0, min(height - target_height, cy - target_height // 2))
    
    else:
        # Default to center crop
        left = (width - target_width) // 2
        top = (height - target_height) // 2
    
    right = left + target_width
    bottom = top + target_height
    
    return (left, top, right, bottom)


def _detect_face_center(image: Image.Image) -> Optional[Tuple[int, int]]:
    """
    Detect face center using MediaPipe Face Detection.
    """
    # Convert PIL Image to OpenCV format (RGB to BGR)
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # type: ignore

    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    ) as face_detection:  # type: ignore
        results: Any = face_detection.process(image_np)  # type: ignore

        if results and hasattr(results, 'detections') and results.detections:
            detections: Any = results.detections
            detection = detections[0]
            if hasattr(detection, 'location_data') and detection.location_data and hasattr(detection.location_data, 'relative_bounding_box'):
                location_data: Any = detection.location_data
                bbox_c = location_data.relative_bounding_box
                ih, iw, _ = image_np.shape
                
                # Convert normalized coordinates to pixel coordinates
                x = int(getattr(bbox_c, 'xmin', 0) * iw)
                y = int(getattr(bbox_c, 'ymin', 0) * ih)
                w = int(getattr(bbox_c, 'width', 0) * iw)
                h = int(getattr(bbox_c, 'height', 0) * ih)
                
                # Calculate center of the bounding box
                cx = x + w // 2
                cy = y + h // 2
                return (cx, cy)
    
    return None


def _detect_salient_region(image: Image.Image) -> Tuple[int, int]:
    """Detect the most salient region in the image."""
    # Convert to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Calculate saliency using spectral residual approach
    saliency = _spectral_residual_saliency(gray.astype(np.uint8))
    
    # Find center of mass of saliency
    height, width = saliency.shape
    y_coords, x_coords = np.ogrid[:height, :width]
    
    total_saliency = np.sum(saliency)
    if total_saliency > 0:
        cx = int(np.sum(x_coords * saliency) / total_saliency)
        cy = int(np.sum(y_coords * saliency) / total_saliency)
    else:
        cx, cy = width // 2, height // 2
    
    return (cx, cy)


def _spectral_residual_saliency(gray: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    """Calculate saliency map using spectral residual method."""
    # Resize for efficiency
    small = cv2.resize(gray, (64, 64))
    
    # Convert to float
    img_float = small.astype(np.float32)
    
    # Compute FFT
    fft = np.fft.fft2(img_float)
    log_amplitude = np.log(np.abs(fft) + 1e-8)
    phase = np.angle(fft)
    
    # Compute spectral residual
    avg_log_amplitude = cv2.blur(log_amplitude, (3, 3))
    spectral_residual = log_amplitude - avg_log_amplitude
    
    # Reconstruct image
    combined = np.exp(spectral_residual + 1j * phase)
    img_back = np.fft.ifft2(combined)
    saliency = np.abs(img_back) ** 2
    
    # Post-process
    saliency = cv2.GaussianBlur(saliency, (9, 9), 2.5)
    saliency = cv2.resize(saliency, (gray.shape[1], gray.shape[0]))
    
    # Normalize
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    return saliency


def _apply_edge_softening(
    image: Image.Image,
    softness: int
) -> Image.Image:
    """Apply soft edges to the image."""
    # Create a mask with soft edges
    width, height = image.size
    
    # Create mask
    mask = Image.new('L', (width, height), 255)
    draw = ImageDraw.Draw(mask)
    
    # Draw gradient borders
    for i in range(softness):
        alpha = int(255 * (i / softness))
        
        # Top edge
        draw.rectangle(
            [(i, i), (width - i - 1, i)],
            fill=alpha
        )
        # Bottom edge
        draw.rectangle(
            [(i, height - i - 1), (width - i - 1, height - i - 1)],
            fill=alpha
        )
        # Left edge
        draw.rectangle(
            [(i, i), (i, height - i - 1)],
            fill=alpha
        )
        # Right edge
        draw.rectangle(
            [(width - i - 1, i), (width - i - 1, height - i - 1)],
            fill=alpha
        )
    
    # Apply Gaussian blur to mask for smoother transition
    mask = mask.filter(ImageFilter.GaussianBlur(radius=softness // 3))
    
    # Create result image with transparency
    result = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    
    # Convert input image to RGBA if needed
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Apply mask
    result.paste(image, (0, 0))
    result.putalpha(mask)
    
    return result


def _blur_periphery(
    image: Image.Image,
    blur_strength: float = 0.5
) -> Image.Image:
    """Apply progressive blur to image periphery."""
    width, height = image.size
    img_array = np.array(image)
    
    # Create radial gradient mask
    center_x, center_y = width // 2, height // 2
    Y, X = np.ogrid[:height, :width]
    
    # Calculate distance from center
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    
    # Normalize distance
    mask = 1 - (dist_from_center / max_dist)
    mask = np.clip(mask, 0, 1)
    
    # Apply sigmoid for smoother transition
    mask = 1 / (1 + np.exp(-10 * (mask - 0.5)))
    
    # Create progressively blurred versions
    blurred_versions = []
    blur_sizes = [3, 7, 11, 15]
    
    for blur_size in blur_sizes:
        if blur_size > 1:
            blurred = cv2.GaussianBlur(img_array, (blur_size, blur_size), 0)
        else:
            blurred = img_array.copy()
        blurred_versions.append(blurred)  # type: ignore
    
    # Combine based on distance from center
    result = np.zeros_like(img_array, dtype=np.float32)
    
    for i, blurred in enumerate(blurred_versions):  # type: ignore
        # Calculate weight for this blur level
        weight_start = i / len(blur_sizes)
        weight_end = (i + 1) / len(blur_sizes)
        
        # Create weight mask for this blur level
        weight = np.zeros_like(mask)
        weight[(mask >= weight_start) & (mask < weight_end)] = 1
        
        if len(img_array.shape) == 3:
            weight = np.stack([weight] * 3, axis=-1)
        
        result += blurred * weight # type: ignore
    
    # Add the sharpest version for the center
    center_weight = mask > (1 - 1/len(blur_sizes))
    if len(img_array.shape) == 3:
        center_weight = np.stack([center_weight] * 3, axis=-1)
    
    result = result * (1 - center_weight) + img_array * center_weight # type: ignore
    
    return Image.fromarray(result.astype(np.uint8))  # type: ignore


def create_vignette(
    image: Image.Image,
    strength: float = 0.5,
    color: Tuple[int, int, int] = (0, 0, 0)
) -> Image.Image:
    """
    Add vignette effect to image edges.
    
    Args:
        image: PIL Image
        strength: Vignette strength (0.0-1.0)
        color: Vignette color (RGB tuple)
    
    Returns:
        Image with vignette effect
    """
    width, height = image.size
    
    # Create vignette mask
    mask = Image.new('L', (width, height), 0)
    
    # Create radial gradient
    for y in range(height):
        for x in range(width):
            # Distance from center
            dx = (x - width / 2) / (width / 2)
            dy = (y - height / 2) / (height / 2)
            distance = math.sqrt(dx**2 + dy**2)
            
            # Calculate vignette intensity
            if distance <= 1:
                intensity = 255 * (1 - distance ** 2) ** (2 * strength)
            else:
                intensity = 0
            
            mask.putpixel((x, y), int(intensity))
    
    # Apply Gaussian blur for smooth transition
    mask = mask.filter(ImageFilter.GaussianBlur(radius=width // 20))
    
    # Create colored overlay
    overlay = Image.new('RGB', (width, height), color)
    
    # Composite images
    result = Image.composite(image, overlay, mask)
    
    return result


def adaptive_crop_for_privacy(
    image: Image.Image,
    sensitivity_map: Optional[npt.NDArray[np.float32]] = None
) -> Image.Image:
    """
    Perform adaptive cropping based on privacy sensitivity map.
    
    Args:
        image: PIL Image to crop
        sensitivity_map: Optional sensitivity map (higher values = more sensitive)
    
    Returns:
        Adaptively cropped image
    """
    width, height = image.size
    
    if sensitivity_map is None:
        # Generate default sensitivity map based on edges and faces
        sensitivity_map = _generate_sensitivity_map(image)
    
    # Find crop region that minimizes sensitive content
    crop_box = _find_privacy_optimal_crop(sensitivity_map, width, height)
    
    # Apply crop
    cropped = image.crop(crop_box)
    
    # Apply additional privacy measures
    cropped = _apply_privacy_blur(cropped, sensitivity_map, crop_box)
    
    return cropped


def _generate_sensitivity_map(image: Image.Image) -> npt.NDArray[np.float32]:
    """Generate a sensitivity map for privacy-aware cropping."""
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # Initialize sensitivity map
    sensitivity = np.zeros((height, width), dtype=np.float32)
    
    # Add face detection sensitivity
    face_center = _detect_face_center(image)
    if face_center:
        cx, cy = face_center
        Y, X = np.ogrid[:height, :width]
        face_mask = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * (width / 4)**2))
        sensitivity += face_mask * 0.5
    
    # Add edge sensitivity (edges often contain identifying features)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    edges = edges.astype(np.float32) / 255.0
    edges = cv2.GaussianBlur(edges, (15, 15), 0)
    sensitivity += edges * 0.3
    
    # Add corner sensitivity
    corners = cv2.goodFeaturesToTrack(
        gray.astype(np.uint8),
        maxCorners=100,
        qualityLevel=0.01,
        minDistance=10
    )
    
    if corners is not None:  # type: ignore
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(sensitivity, (int(x), int(y)), 20, (1.0,), -1)

    # Normalize
    sensitivity = np.clip(sensitivity, 0, 1)
    
    return sensitivity


def _find_privacy_optimal_crop(
    sensitivity_map: npt.NDArray[np.float32],
    target_width: int,
    target_height: int
) -> Tuple[int, int, int, int]:
    """Find crop region that minimizes sensitive content."""
    map_height, map_width = sensitivity_map.shape
    
    # Ensure we don't crop more than available
    crop_width = min(target_width, map_width)
    crop_height = min(target_height, map_height)
    
    # Use sliding window to find region with minimum sensitivity
    min_sensitivity = float('inf')
    best_crop = (0, 0, crop_width, crop_height)
    
    # Slide window with stride
    stride = 10
    for y in range(0, map_height - crop_height + 1, stride):
        for x in range(0, map_width - crop_width + 1, stride):
            # Calculate average sensitivity in this region
            region = sensitivity_map[y:y+crop_height, x:x+crop_width]
            avg_sensitivity = np.mean(region)
            
            if avg_sensitivity < min_sensitivity:
                min_sensitivity = avg_sensitivity
                best_crop = (x, y, x + crop_width, y + crop_height)
    
    return best_crop


def _apply_privacy_blur(
    image: Image.Image,
    sensitivity_map: npt.NDArray[np.float32],
    crop_box: Tuple[int, int, int, int]
) -> Image.Image:
    """Apply selective blur based on sensitivity within cropped region."""
    # Extract relevant portion of sensitivity map
    x1, y1, x2, y2 = crop_box
    width, height = image.size
    
    # Resize sensitivity map to match cropped image
    cropped_sensitivity = sensitivity_map[y1:y2, x1:x2]
    cropped_sensitivity = cv2.resize(
        cropped_sensitivity,
        (width, height),
        interpolation=cv2.INTER_LINEAR
    )  # type: ignore
    
    # Apply selective blur
    img_array = np.array(image)
    blurred = cv2.GaussianBlur(img_array, (21, 21), 0)  # type: ignore
    
    # Blend based on sensitivity
    if len(img_array.shape) == 3:
        sensitivity_3d = np.stack([cropped_sensitivity] * 3, axis=-1)  # type: ignore
    else:
        sensitivity_3d = cropped_sensitivity
    
    result = img_array * (1 - sensitivity_3d * 0.5) + blurred * (sensitivity_3d * 0.5)  # type: ignore
    
    return Image.fromarray(result.astype(np.uint8))  # type: ignore
