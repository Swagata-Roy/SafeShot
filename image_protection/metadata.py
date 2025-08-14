"""
Metadata protection module for EXIF stripping and watermarking.

This module handles metadata removal and watermark addition to protect
image ownership and privacy.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ExifTags
import io
from typing import Optional, Tuple, Dict, Any
import datetime
import piexif


def strip_metadata(image: Image.Image) -> Image.Image:
    """
    Remove all EXIF and metadata from an image using piexif.
    
    Args:
        image: PIL Image to process
    
    Returns:
        PIL Image with metadata stripped
    """
    try:
        # Load EXIF data
        exif_bytes = piexif.dump({})
        
        # Save image without EXIF data
        img_byte_arr = io.BytesIO()
        # Handle RGBA images appropriately
        img_to_save = image
        format_to_use = image.format or "JPEG"
        if image.mode == "RGBA":
            if format_to_use == "JPEG":
                # Convert to RGB for JPEG
                img_to_save = Image.new("RGB", image.size, (255, 255, 255))
                img_to_save.paste(image, mask=image.split()[3])
            else:
                # Keep RGBA for formats that support it
                pass
        img_to_save.save(img_byte_arr, format=format_to_use, exif=exif_bytes)
        img_byte_arr.seek(0)
        return Image.open(img_byte_arr)
    except Exception as e:
        # Fallback to original method if piexif fails
        print(f"Piexif stripping failed: {e}. Falling back to basic stripping.")
        data = list(image.getdata())
        image_without_exif = Image.new(image.mode, image.size)
        image_without_exif.putdata(data)
        
        if hasattr(image, 'info'):
            safe_info = {}
            for key, value in image.info.items():
                if isinstance(key, str) and key.lower() in ['dpi', 'transparency']:
                    safe_info[key] = value
            for key, value in safe_info.items():
                image_without_exif.info[key] = value
        return image_without_exif


def get_metadata_info(image: Image.Image) -> Dict[str, Any]:
    """
    Extract and return metadata information from an image using piexif.
    
    Args:
        image: PIL Image to analyze
    
    Returns:
        Dictionary containing metadata information
    """
    metadata: Dict[str, Any] = {
        'exif': {},
        'info': {},
        'format': image.format,
        'mode': image.mode,
        'size': image.size
    }
    
    # Extract EXIF data using piexif
    try:
        exif_dict = piexif.load(image.info.get("exif", b""))
        for ifd_name in exif_dict:
            if ifd_name == "thumbnail":
                continue
            for key, value in exif_dict[ifd_name].items():
                try:
                    tag_name = piexif.TAGS.get(ifd_name, {}).get(key, {}).get("name", str(key))
                except KeyError:
                    tag_name = str(key)
                metadata['exif'][str(tag_name)] = str(value)
    except Exception:
        # Fallback to PIL's getexif if piexif fails or no exif data
        try:
            exifdata = image.getexif()
            if exifdata:
                for tag_id, value in exifdata.items():
                    tag = ExifTags.TAGS.get(tag_id, str(tag_id))
                    metadata['exif'][str(tag)] = str(value)
        except:
            pass
    
    # Extract other metadata
    if hasattr(image, 'info'):
        metadata['info'] = dict(image.info)
    
    return metadata


def add_watermark(
    image: Image.Image,
    text: str,
    opacity: float = 0.3,
    position: str = "bottom-right",
    font_size: int = 50,
    color: Tuple[int, int, int] = (255, 255, 255),
    shadow: bool = True,
    font_path: str = "assets/Classica.ttf"
) -> Image.Image:
    """
    Add a text watermark to an image.
    
    Args:
        image: PIL Image to watermark
        text: Watermark text
        opacity: Watermark opacity (0.0 to 1.0)
        position: Position of watermark ("bottom-right", "bottom-left", "top-right", "top-left", "center")
        font_size: Font size in pixels.
        color: RGB color tuple for watermark
        shadow: Whether to add shadow for better visibility
        font_path: Path to the font file to use.
    
    Returns:
        Watermarked PIL Image
    """
    # Convert to RGBA for transparency support
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Create a transparent overlay
    width, height = image.size
    watermark = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(watermark)
    
    # Load the font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        # Fallback to a basic default font if the specified font is not found
        font = ImageFont.load_default()
    
    # Get text dimensions
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Calculate position
    margin = 20
    if position == "bottom-right":
        x = width - text_width - margin
        y = height - text_height - margin
    elif position == "bottom-left":
        x = margin
        y = height - text_height - margin
    elif position == "top-right":
        x = width - text_width - margin
        y = margin
    elif position == "top-left":
        x = margin
        y = margin
    elif position == "center":
        x = (width - text_width) // 2
        y = (height - text_height) // 2
    else:
        # Default to bottom-right
        x = width - text_width - margin
        y = height - text_height - margin
    
    # Draw shadow if requested
    if shadow:
        shadow_offset = max(1, font_size // 20)
        shadow_color = (0, 0, 0, int(255 * opacity * 0.7))
        draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=shadow_color)
    
    # Draw main text
    text_color = color + (int(255 * opacity),)
    draw.text((x, y), text, font=font, fill=text_color)
    
    # Composite the watermark with the image
    watermarked = Image.alpha_composite(image, watermark)
    
    # Convert back to original mode if needed
    if watermarked.mode == 'RGBA' and image.mode != 'RGBA':
        # Create a white background
        background = Image.new('RGB', watermarked.size, (255, 255, 255))
        background.paste(watermarked, mask=watermarked.split()[3])
        watermarked = background
    
    return watermarked


def add_image_watermark(
    image: Image.Image,
    watermark_image: Image.Image,
    opacity: float = 0.3,
    position: str = "bottom-right",
    scale: float = 0.2
) -> Image.Image:
    """
    Add an image watermark to another image.
    
    Args:
        image: PIL Image to watermark
        watermark_image: PIL Image to use as watermark
        opacity: Watermark opacity (0.0 to 1.0)
        position: Position of watermark
        scale: Scale of watermark relative to main image
    
    Returns:
        Watermarked PIL Image
    """
    # Convert to RGBA
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Prepare watermark
    watermark = watermark_image.convert('RGBA')
    
    # Scale watermark
    width, height = image.size
    watermark_width = int(width * scale)
    watermark_height = int(watermark.height * (watermark_width / watermark.width))
    watermark = watermark.resize((watermark_width, watermark_height), Image.Resampling.LANCZOS)
    
    # Adjust opacity
    watermark_array = np.array(watermark)
    watermark_array[:, :, 3] = (watermark_array[:, :, 3] * opacity).astype(np.uint8)
    watermark = Image.fromarray(watermark_array)
    
    # Calculate position
    margin = 20
    if position == "bottom-right":
        x = width - watermark_width - margin
        y = height - watermark_height - margin
    elif position == "bottom-left":
        x = margin
        y = height - watermark_height - margin
    elif position == "top-right":
        x = width - watermark_width - margin
        y = margin
    elif position == "top-left":
        x = margin
        y = margin
    elif position == "center":
        x = (width - watermark_width) // 2
        y = (height - watermark_height) // 2
    else:
        x = width - watermark_width - margin
        y = height - watermark_height - margin
    
    # Create a new image for compositing
    result = image.copy()
    result.paste(watermark, (x, y), watermark)
    
    return result


def add_invisible_watermark(
    image: Image.Image,
    signature: str,
    strength: float = 0.01
) -> Image.Image:
    """
    Add an invisible watermark using LSB steganography.
    
    Args:
        image: PIL Image to watermark
        signature: Text signature to embed
        strength: Embedding strength (very low for invisibility)
    
    Returns:
        Image with invisible watermark
    """
    # Convert to numpy array
    img_array = np.array(image)
    _, _ = img_array.shape[:2]  # height, width
    
    # Convert signature to binary
    binary_signature = ''.join(format(ord(char), '08b') for char in signature)
    
    # Add length information
    signature_length = len(binary_signature)
    length_binary = format(signature_length, '032b')
    full_binary = length_binary + binary_signature
    
    # Embed in least significant bits
    flat_array = img_array.flatten()
    
    for i, bit in enumerate(full_binary):
        if i < len(flat_array):
            # Modify LSB
            if bit == '1':
                flat_array[i] = flat_array[i] | 1
            else:
                flat_array[i] = flat_array[i] & ~1
    
    watermarked_array = flat_array.reshape(img_array.shape)
    
    return Image.fromarray(watermarked_array.astype(np.uint8))


def extract_invisible_watermark(image: Image.Image) -> Optional[str]:
    """
    Extract invisible watermark from an image.
    
    Args:
        image: PIL Image potentially containing watermark
    
    Returns:
        Extracted signature or None if not found
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # Extract from least significant bits
    if len(img_array.shape) == 3:
        flat_array = img_array.flatten()
    else:
        flat_array = img_array.flatten()
    
    # Extract length first (32 bits)
    length_binary = ''
    for i in range(32):
        if i < len(flat_array):
            length_binary += str(flat_array[i] & 1)
    
    try:
        signature_length = int(length_binary, 2)
        
        # Sanity check
        if signature_length <= 0 or signature_length > 10000:
            return None
        
        # Extract signature
        signature_binary = ''
        for i in range(32, 32 + signature_length):
            if i < len(flat_array):
                signature_binary += str(flat_array[i] & 1)
        
        # Convert binary to text
        signature = ''
        for i in range(0, len(signature_binary), 8):
            byte = signature_binary[i:i+8]
            if len(byte) == 8:
                signature += chr(int(byte, 2))
        
        return signature
    except:
        return None


def add_copyright_banner(
    image: Image.Image,
    copyright_text: str,
    background_color: Tuple[int, int, int] = (0, 0, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    position: str = "bottom"
) -> Image.Image:
    """
    Add a copyright banner to the image.
    
    Args:
        image: PIL Image
        copyright_text: Copyright text to display
        background_color: RGB color for banner background
        text_color: RGB color for text
        position: Banner position ("top" or "bottom")
    
    Returns:
        Image with copyright banner
    """
    width, height = image.size
    
    # Calculate banner height
    banner_height = max(30, height // 20)
    
    # Create new image with space for banner
    if position == "bottom":
        new_height = height + banner_height
        new_image = Image.new(image.mode, (width, new_height), background_color)
        new_image.paste(image, (0, 0))
        banner_y = height
    else:  # top
        new_height = height + banner_height
        new_image = Image.new(image.mode, (width, new_height), background_color)
        new_image.paste(image, (0, banner_height))
        banner_y = 0
    
    # Draw copyright text
    draw = ImageDraw.Draw(new_image)
    
    # Try to load font
    try:
        font_size = banner_height - 10
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Get text dimensions
    bbox = draw.textbbox((0, 0), copyright_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center text in banner
    text_x = (width - text_width) // 2
    text_y = banner_y + (banner_height - text_height) // 2
    
    # Draw text
    draw.text((text_x, text_y), copyright_text, font=font, fill=text_color)
    
    return new_image


def add_timestamp_watermark(
    image: Image.Image,
    format_string: str = "%Y-%m-%d %H:%M:%S",
    **kwargs: Any
) -> Image.Image:
    """
    Add a timestamp watermark to the image.
    
    Args:
        image: PIL Image
        format_string: DateTime format string
        **kwargs: Additional arguments passed to add_watermark
    
    Returns:
        Image with timestamp watermark
    """
    timestamp = datetime.datetime.now().strftime(format_string)
    return add_watermark(image, timestamp, **kwargs)


def create_watermark_pattern(
    width: int,
    height: int,
    text: str,
    spacing: int = 100,
    angle: float = -45,
    opacity: float = 0.1
) -> Image.Image:
    """
    Create a repeating watermark pattern.
    
    Args:
        width: Pattern width
        height: Pattern height
        text: Watermark text
        spacing: Spacing between repetitions
        angle: Rotation angle in degrees
        opacity: Pattern opacity
    
    Returns:
        Watermark pattern as PIL Image
    """
    # Create transparent image
    pattern = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(pattern)
    
    # Load font
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Calculate text dimensions
    bbox = draw.textbbox((0, 0), text, font=font)
    _ = bbox[2] - bbox[0]  # text_width
    _ = bbox[3] - bbox[1]  # text_height
    
    # Create rotated text pattern
    temp_size = int(np.sqrt(width**2 + height**2))
    temp_pattern = Image.new('RGBA', (temp_size, temp_size), (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp_pattern)
    
    # Draw repeating text
    for y in range(-temp_size, temp_size * 2, spacing):
        for x in range(-temp_size, temp_size * 2, spacing):
            temp_draw.text(
                (x, y),
                text,
                font=font,
                fill=(128, 128, 128, int(255 * opacity))
            )
    
    # Rotate pattern
    temp_pattern = temp_pattern.rotate(angle, expand=1)
    
    # Crop to desired size
    left = (temp_pattern.width - width) // 2
    top = (temp_pattern.height - height) // 2
    pattern = temp_pattern.crop((left, top, left + width, top + height))
    
    return pattern


def apply_pattern_watermark(
    image: Image.Image,
    text: str,
    spacing: int = 150,
    angle: float = -45,
    opacity: float = 0.1
) -> Image.Image:
    """
    Apply a repeating pattern watermark to the entire image.
    
    Args:
        image: PIL Image
        text: Watermark text
        spacing: Spacing between repetitions
        angle: Rotation angle
        opacity: Watermark opacity
    
    Returns:
        Watermarked image
    """
    # Convert to RGBA
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Create pattern
    pattern = create_watermark_pattern(
        image.width,
        image.height,
        text,
        spacing,
        angle,
        opacity
    )
    
    # Composite
    watermarked = Image.alpha_composite(image, pattern)
    
    # Convert back if needed
    if image.mode != 'RGBA':
        background = Image.new('RGB', watermarked.size, (255, 255, 255))
        background.paste(watermarked, mask=watermarked.split()[3])
        watermarked = background
    
    return watermarked
