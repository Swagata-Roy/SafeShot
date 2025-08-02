"""
Utility functions for image processing and protection.

This module contains reusable helper functions for various image operations.
"""

from typing import Tuple, List, Optional, Dict, cast, Any
import numpy as np
import numpy.typing as npt
from PIL import Image
import cv2
import base64
import io
import imageio.v3 as iio
import mediapipe as mp

# Type aliases
NDArray = npt.NDArray[np.float64]


def resize_image(
    image: Image.Image,
    max_size: Optional[Tuple[int, int]] = None,
    maintain_aspect: bool = True
) -> Image.Image:
    """
    Resize an image while optionally maintaining aspect ratio.
    
    Args:
        image: PIL Image to resize
        max_size: Maximum (width, height) tuple
        maintain_aspect: Whether to maintain aspect ratio
    
    Returns:
        Resized PIL Image
    """
    if max_size is None:
        return image
    
    width, height = image.size
    max_width, max_height = max_size
    
    if maintain_aspect:
        # Calculate scaling factor
        scale = min(max_width / width, max_height / height)
        
        # Only resize if image is larger than max_size
        if scale < 1:
            new_width = int(width * scale)
            new_height = int(height * scale)
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    else:
        return image.resize(max_size, Image.Resampling.LANCZOS)
    
    return image


def normalize_image(image: Image.Image) -> Image.Image:
    """
    Normalize image to standard format and color space.
    
    Args:
        image: PIL Image
    
    Returns:
        Normalized PIL Image
    """
    # Convert to RGB if necessary
    if image.mode not in ['RGB', 'RGBA']:
        if image.mode == 'P':
            # Convert palette images to RGBA first to preserve transparency
            image = image.convert('RGBA')
        elif image.mode in ['L', 'LA']:
            # Convert grayscale to RGB
            image = image.convert('RGB')
        else:
            # For other modes, convert to RGB
            image = image.convert('RGB')
    
    return image


def calculate_image_hash(
    image: Image.Image,
    hash_type: str = "average"
) -> str:
    """
    Calculate perceptual hash of an image.
    
    Args:
        image: PIL Image
        hash_type: Type of hash ("average", "difference", "perceptual")
    
    Returns:
        Hash string
    """
    # Resize image to standard size for hashing
    img = image.convert('L').resize((8, 8), Image.Resampling.LANCZOS)
    pixels = np.array(img)
    
    if hash_type == "average":
        # Average hash
        avg = pixels.mean()
        hash_bits = (pixels > avg).flatten()
    elif hash_type == "difference":
        # Difference hash
        diff = pixels[:, 1:] > pixels[:, :-1]
        hash_bits = diff.flatten()
    else:
        # Simple perceptual hash
        avg = pixels.mean()
        hash_bits = (pixels > avg).flatten()
    
    # Convert to hex string
    hash_int = 0
    for bit in hash_bits:
        hash_int = (hash_int << 1) | int(bit)
    
    return hex(hash_int)[2:].zfill(16)


def compare_images(
    image1: Image.Image,
    image2: Image.Image,
    method: str = "mse"
) -> float:
    """
    Compare two images using various metrics.
    
    Args:
        image1: First PIL Image
        image2: Second PIL Image
        method: Comparison method ("mse", "ssim", "hash")
    
    Returns:
        Similarity score (higher means more similar)
    """
    # Ensure images are same size
    if image1.size != image2.size:
        image2 = image2.resize(image1.size, Image.Resampling.LANCZOS)
    
    # Convert to numpy arrays
    arr1 = np.array(image1)
    arr2 = np.array(image2)
    
    if method == "mse":
        # Mean Squared Error (lower is better, so we invert)
        mse = np.mean((arr1 - arr2) ** 2)
        return float(1.0 / (1.0 + mse))  # Explicitly cast to Python float
    
    elif method == "ssim":
        # Structural Similarity Index
        return calculate_ssim(arr1, arr2)
    
    elif method == "hash":
        # Hash comparison
        hash1 = calculate_image_hash(image1)
        hash2 = calculate_image_hash(image2)
        
        # Calculate Hamming distance
        distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
        return 1.0 - (distance / len(hash1))
    
    else:
        raise ValueError(f"Unknown comparison method: {method}")


def calculate_ssim(img1: NDArray, img2: NDArray) -> float:
    """
    Calculate Structural Similarity Index between two images.
    
    Args:
        img1: First image array
        img2: Second image array
    
    Returns:
        SSIM value (0-1, higher is more similar)
    """
    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        img1 = cast(NDArray, cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY))
    if len(img2.shape) == 3:
        img2 = cast(NDArray, cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY))
    
    # Constants for SSIM
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Calculate means
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Calculate variances and covariance
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    
    # SSIM calculation
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return float(np.mean(ssim_map))  # Explicitly cast to Python float


def blend_images(
    image1: Image.Image,
    image2: Image.Image,
    alpha: float = 0.5,
    blend_mode: str = "normal"
) -> Image.Image:
    """
    Blend two images together.
    
    Args:
        image1: First PIL Image
        image2: Second PIL Image
        alpha: Blend factor (0-1)
        blend_mode: Blending mode ("normal", "multiply", "screen", "overlay")
    
    Returns:
        Blended PIL Image
    """
    # Ensure same size
    if image1.size != image2.size:
        image2 = image2.resize(image1.size, Image.Resampling.LANCZOS)
    
    # Convert to RGBA
    img1 = image1.convert('RGBA')
    img2 = image2.convert('RGBA')
    
    # Convert to arrays
    arr1 = np.array(img1).astype(np.float32) / 255.0
    arr2 = np.array(img2).astype(np.float32) / 255.0
    
    if blend_mode == "normal":
        result = arr1 * (1 - alpha) + arr2 * alpha
    
    elif blend_mode == "multiply":
        result = arr1 * arr2
        result = arr1 * (1 - alpha) + result * alpha
    
    elif blend_mode == "screen":
        result = 1 - (1 - arr1) * (1 - arr2)
        result = arr1 * (1 - alpha) + result * alpha
    
    elif blend_mode == "overlay":
        # Overlay blend mode
        mask = arr1 < 0.5
        result = np.zeros_like(arr1)
        result[mask] = 2 * arr1[mask] * arr2[mask]
        result[~mask] = 1 - 2 * (1 - arr1[~mask]) * (1 - arr2[~mask])
        result = arr1 * (1 - alpha) + result * alpha
    
    else:
        raise ValueError(f"Unknown blend mode: {blend_mode}")
    
    # Convert back to image
    result = (result * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(result, 'RGBA')


def create_noise_pattern(
    size: Tuple[int, int],
    noise_type: str = "gaussian",
    intensity: float = 0.1
) -> NDArray:
    """
    Create various noise patterns.
    
    Args:
        size: (width, height) tuple
        noise_type: Type of noise ("gaussian", "uniform", "salt_pepper")
        intensity: Noise intensity
    
    Returns:
        Noise array
    """
    width, height = size
    
    if noise_type == "gaussian":
        noise = np.random.randn(height, width, 3) * intensity
    
    elif noise_type == "uniform":
        noise = (np.random.rand(height, width, 3) - 0.5) * 2 * intensity
    
    elif noise_type == "salt_pepper":
        noise = np.zeros((height, width, 3))
        # Salt
        salt_mask = np.random.rand(height, width) < intensity / 2
        noise[salt_mask] = 1
        # Pepper
        pepper_mask = np.random.rand(height, width) < intensity / 2
        noise[pepper_mask] = -1
    
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return noise


def detect_image_format(image_data: bytes) -> str:
    """
    Detect image format from bytes.
    
    Args:
        image_data: Image data as bytes
    
    Returns:
        Format string (e.g., "JPEG", "PNG")
    """
    # Check magic numbers
    if image_data.startswith(b'\xff\xd8\xff'):
        return "JPEG"
    elif image_data.startswith(b'\x89PNG'):
        return "PNG"
    elif image_data.startswith(b'GIF87a') or image_data.startswith(b'GIF89a'):
        return "GIF"
    elif image_data.startswith(b'BM'):
        return "BMP"
    elif image_data.startswith(b'II\x2a\x00') or image_data.startswith(b'MM\x00\x2a'):
        return "TIFF"
    elif image_data.startswith(b'RIFF') and image_data[8:12] == b'WEBP':
        return "WEBP"
    else:
        # Try to detect with imageio
        try:
            with io.BytesIO(image_data) as buffer:
                img_array = iio.imread(buffer)
                format = getattr(img_array, 'meta', {}).get('format', 'UNKNOWN')
                return format.upper() if format != 'UNKNOWN' else "UNKNOWN"
        except Exception:
            return "UNKNOWN"

def load_image(file_path: str) -> Image.Image:
    """
    Load an image from file with extended format support using imageio.

    Args:
        file_path: Path to image file

    Returns:
        PIL Image
    """
    try:
        # Try to read with imageio first (supports more formats)
        img_array = iio.imread(file_path)
        return Image.fromarray(img_array)
    except Exception:
        # Fall back to PIL for formats imageio doesn't handle
        return Image.open(file_path)

def save_image(image: Image.Image, file_path: str, format: Optional[str] = None) -> None:
    """
    Save an image with extended format support using imageio.

    Args:
        image: PIL Image to save
        file_path: Destination path
        format: Optional format override
    """
    try:
        # Convert to array for imageio
        img_array = np.array(image)

        # Determine format if not specified
        if format is None:
            format = file_path.split('.')[-1].upper()

        # Try to write with imageio first
        iio.imwrite(file_path, img_array, format=format)
    except Exception:
        # Fall back to PIL
        image.save(file_path, format=format)


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    Convert PIL Image to base64 string.
    
    Args:
        image: PIL Image
        format: Output format
    
    Returns:
        Base64 encoded string
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"


def base64_to_image(base64_str: str) -> Image.Image:
    """
    Convert base64 string to PIL Image.
    
    Args:
        base64_str: Base64 encoded image string
    
    Returns:
        PIL Image
    """
    # Remove data URL prefix if present
    if base64_str.startswith('data:'):
        base64_str = base64_str.split(',')[1]
    
    img_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_data))


def calculate_image_entropy(image: Image.Image) -> float:
    """
    Calculate entropy of an image (measure of information content).
    
    Args:
        image: PIL Image
    
    Returns:
        Entropy value
    """
    # Convert to grayscale
    gray = image.convert('L')
    
    # Get histogram
    histogram = gray.histogram()
    
    # Calculate probabilities
    total_pixels = sum(histogram)
    probabilities = [h / total_pixels for h in histogram if h > 0]
    
    # Calculate entropy
    entropy = -sum(p * np.log2(p) for p in probabilities)
    
    return entropy


def apply_color_adjustment(
    image: Image.Image,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    hue_shift: float = 0.0
) -> Image.Image:
    """
    Apply color adjustments to an image.
    
    Args:
        image: PIL Image
        brightness: Brightness multiplier
        contrast: Contrast multiplier
        saturation: Saturation multiplier
        hue_shift: Hue shift in degrees
    
    Returns:
        Adjusted PIL Image
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # Apply brightness
    if brightness != 1.0:
        img_array = img_array * brightness
        img_array = np.clip(img_array, 0, 255)
    
    # Apply contrast
    if contrast != 1.0:
        mean = img_array.mean()
        img_array = (img_array - mean) * contrast + mean
        img_array = np.clip(img_array, 0, 255)
    
    # Apply saturation and hue shift in HSV space
    if saturation != 1.0 or hue_shift != 0.0:
        hsv = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Apply hue shift
        if hue_shift != 0.0:
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        
        # Apply saturation
        if saturation != 1.0:
            hsv[:, :, 1] = hsv[:, :, 1] * saturation
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        img_array = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    return Image.fromarray(img_array.astype(np.uint8))


def create_gradient(
    size: Tuple[int, int],
    colors: List[Tuple[int, int, int]],
    direction: str = "horizontal"
) -> Image.Image:
    """
    Create a gradient image.
    
    Args:
        size: (width, height) tuple
        colors: List of RGB color tuples
        direction: Gradient direction ("horizontal", "vertical", "diagonal", "radial")
    
    Returns:
        Gradient PIL Image
    """
    width, height = size
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    
    if len(colors) < 2:
        raise ValueError("Need at least 2 colors for gradient")
    
    if direction == "horizontal":
        for i in range(width):
            ratio = i / (width - 1)
            color_idx = ratio * (len(colors) - 1)
            idx1 = int(color_idx)
            idx2 = min(idx1 + 1, len(colors) - 1)
            
            local_ratio = color_idx - idx1
            color = [
                int(colors[idx1][j] * (1 - local_ratio) + colors[idx2][j] * local_ratio)
                for j in range(3)
            ]
            gradient[:, i] = color
    
    elif direction == "vertical":
        for i in range(height):
            ratio = i / (height - 1)
            color_idx = ratio * (len(colors) - 1)
            idx1 = int(color_idx)
            idx2 = min(idx1 + 1, len(colors) - 1)
            
            local_ratio = color_idx - idx1
            color = [
                int(colors[idx1][j] * (1 - local_ratio) + colors[idx2][j] * local_ratio)
                for j in range(3)
            ]
            gradient[i, :] = color
    
    elif direction == "diagonal":
        for y in range(height):
            for x in range(width):
                ratio = (x + y) / (width + height - 2)
                color_idx = ratio * (len(colors) - 1)
                idx1 = int(color_idx)
                idx2 = min(idx1 + 1, len(colors) - 1)
                
                local_ratio = color_idx - idx1
                color = [
                    int(colors[idx1][j] * (1 - local_ratio) + colors[idx2][j] * local_ratio)
                    for j in range(3)
                ]
                gradient[y, x] = color
    
    elif direction == "radial":
        center_x, center_y = width // 2, height // 2
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        for y in range(height):
            for x in range(width):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                ratio = min(dist / max_dist, 1.0)
                color_idx = ratio * (len(colors) - 1)
                idx1 = int(color_idx)
                idx2 = min(idx1 + 1, len(colors) - 1)
                
                local_ratio = color_idx - idx1
                color = [
                    int(colors[idx1][j] * (1 - local_ratio) + colors[idx2][j] * local_ratio)
                    for j in range(3)
                ]
                gradient[y, x] = color
    
    return Image.fromarray(gradient)


def estimate_quality(image: Image.Image) -> Dict[str, float]:
    """
    Estimate image quality metrics.
    
    Args:
        image: PIL Image
    
    Returns:
        Dictionary with quality metrics
    """
    # Convert to array
    img_array = np.array(image)
    
    # Calculate sharpness using Laplacian variance
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    
    # Calculate noise estimate
    noise = estimate_noise(cast(NDArray, gray))
    
    # Calculate contrast
    contrast = gray.std()
    
    # Calculate brightness
    brightness = gray.mean()
    
    # Calculate entropy
    entropy = calculate_image_entropy(image)
    
    return {
        "sharpness": float(sharpness),
        "noise": float(noise),
        "contrast": float(contrast),
        "brightness": float(brightness),
        "entropy": float(entropy),
        "quality_score": calculate_quality_score(sharpness, noise, contrast)
    }


def estimate_noise(gray: NDArray) -> float:
    """
    Estimate noise level in grayscale image.
    
    Args:
        gray: Grayscale image array (will be converted to float64)
    
    Returns:
        Noise estimate as Python float
    """
    # Use median absolute deviation
    median = np.median(gray)
    mad = np.median(np.abs(gray - median))
    noise = mad * 1.4826  # Scale factor for Gaussian noise
    
    return float(noise)  # Explicitly cast numpy.float64 to Python float


def calculate_quality_score(sharpness: float, noise: float, contrast: float) -> float:
    """
    Calculate overall quality score.
    
    Args:
        sharpness: Sharpness value
        noise: Noise value
        contrast: Contrast value
    
    Returns:
        Quality score (0-100)
    """
    # Normalize values
    sharpness_score = min(sharpness / 1000, 1.0) * 40
    noise_score = max(0, 1 - noise / 50) * 30
    contrast_score = min(contrast / 50, 1.0) * 30
    
    return sharpness_score + noise_score + contrast_score
def detect_faces(image: Image.Image) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces in an image using MediaPipe Face Detection.
    
    Args:
        image: PIL Image
    
    Returns:
        List of face bounding boxes (x, y, w, h)
    """
    mp_face_detection = mp.solutions.face_detection
    
    image_np = np.array(image.convert('RGB'))
    
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    ) as face_detection:
        results = face_detection.process(image_np)
        
        face_boxes = []
        if results.detections:
            for detection in results.detections:
                bbox_c = detection.location_data.relative_bounding_box
                ih, iw, _ = image_np.shape
                x, y, w, h = (
                    int(bbox_c.xmin * iw),
                    int(bbox_c.ymin * ih),
                    int(bbox_c.width * iw),
                    int(bbox_c.height * ih),
                )
                face_boxes.append((x, y, w, h))
        return face_boxes