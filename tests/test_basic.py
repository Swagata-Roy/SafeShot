"""
Basic tests for SafeShot image protection functionality.

This module contains functional and integration tests for the SafeShot
image protection system, ensuring all protection methods work correctly.
"""

import unittest
from PIL import Image
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_protection.cloak import apply_cloaking
from image_protection.style_defense import apply_style_defense
from image_protection.cropper import smart_crop, privacy_crop
from image_protection.metadata import strip_metadata, add_watermark
from image_protection.utils import normalize_image, resize_image
from image_protection.version import get_version, get_protection_methods


class TestSafeShot(unittest.TestCase):
    """Test suite for SafeShot image protection functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test image
        self.test_image = Image.new('RGB', (100, 100), color='red')
        self.test_image_path = 'test_temp.png'
        self.test_image.save(self.test_image_path)

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)

    def test_version_info(self):
        """Test version information retrieval."""
        version = get_version()
        self.assertIsInstance(version, str)
        self.assertTrue(len(version) > 0)

    def test_protection_methods(self):
        """Test available protection methods."""
        methods = get_protection_methods()
        self.assertIsInstance(methods, list)
        self.assertTrue(len(methods) > 0)
        self.assertIn('cloak', methods)
        self.assertIn('style_defense', methods)
        self.assertIn('cropper', methods)
        self.assertIn('metadata', methods)

    def test_image_utils(self):
        """Test image utility functions."""
        # Test normalizing image
        normalized = normalize_image(self.test_image)
        self.assertIsInstance(normalized, Image.Image)
        self.assertEqual(normalized.mode, 'RGB')

        # Test resizing image
        resized = resize_image(self.test_image, (50, 50))
        self.assertIsInstance(resized, Image.Image)
        self.assertEqual(resized.size, (50, 50))

    def test_cloaking_functionality(self):
        """Test AI cloaking protection."""
        # Test with valid image
        result = apply_cloaking(self.test_image, intensity=0.5)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.test_image.size)

        # Test with different intensity levels
        for intensity in [0.1, 0.5, 0.9]:
            result = apply_cloaking(self.test_image, intensity=intensity)
            self.assertIsInstance(result, Image.Image)

    def test_style_defense_functionality(self):
        """Test style defense protection."""
        # Test with valid image
        result = apply_style_defense(self.test_image, strength=0.3)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.test_image.size)

        # Test with different strength levels
        for strength in [0.1, 0.5, 0.9]:
            result = apply_style_defense(self.test_image, strength=strength)
            self.assertIsInstance(result, Image.Image)

    def test_smart_crop_functionality(self):
        """Test smart cropping functionality."""
        # Test with valid image
        result = smart_crop(self.test_image, aspect_ratio="1:1", edge_softness=10)
        self.assertIsInstance(result, Image.Image)
        self.assertTrue(result.size[0] <= self.test_image.size[0])
        self.assertTrue(result.size[1] <= self.test_image.size[1])

        # Test different aspect ratios
        for ratio in ["1:1", "4:3", "16:9"]:
            result = smart_crop(self.test_image, aspect_ratio=ratio)
            self.assertIsInstance(result, Image.Image)

    def test_privacy_crop_functionality(self):
        """Test privacy-focused cropping."""
        # Test with valid image
        result = privacy_crop(self.test_image, remove_edges=True, blur_periphery=True)
        self.assertIsInstance(result, Image.Image)
        self.assertTrue(result.size[0] <= self.test_image.size[0])
        self.assertTrue(result.size[1] <= self.test_image.size[1])

    def test_metadata_scrubbing(self):
        """Test metadata stripping functionality."""
        # Create image with some metadata
        test_image_with_meta = Image.new('RGB', (100, 100), color='blue')
        
        # Test metadata stripping
        result = strip_metadata(test_image_with_meta)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, test_image_with_meta.size)

    def test_watermark_addition(self):
        """Test watermark addition functionality."""
        # Test with valid image
        result = add_watermark(self.test_image, text="Test Watermark")
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.test_image.size)

        # Test with different watermark texts
        for text in ["Test", "Protected", "SafeShot"]:
            result = add_watermark(self.test_image, text=text)
            self.assertIsInstance(result, Image.Image)

    def test_image_saving(self):
        """Test image saving functionality."""
        output_path = 'test_output.png'
        
        # Test saving image directly with PIL
        self.test_image.save(output_path)
        self.assertTrue(os.path.exists(output_path))
        
        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)

    def test_integration_pipeline(self):
        """Test complete protection pipeline."""
        # Apply multiple protections in sequence
        image = self.test_image
        
        # Step 1: Apply cloaking
        image = apply_cloaking(image, intensity=0.3)
        self.assertIsInstance(image, Image.Image)
        
        # Step 2: Apply style defense
        image = apply_style_defense(image, strength=0.2)
        self.assertIsInstance(image, Image.Image)
        
        # Step 3: Apply smart crop
        image = smart_crop(image, aspect_ratio="1:1")
        self.assertIsInstance(image, Image.Image)
        
        # Step 4: Scrub metadata
        image = strip_metadata(image)
        self.assertIsInstance(image, Image.Image)
        
        # Step 5: Add watermark
        image = add_watermark(image, text="Protected")
        self.assertIsInstance(image, Image.Image)

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with very small image
        small_image = Image.new('RGB', (10, 10), color='green')
        result = apply_cloaking(small_image, intensity=0.5)
        self.assertIsInstance(result, Image.Image)

        # Test with large image
        large_image = Image.new('RGB', (1000, 1000), color='yellow')
        result = apply_style_defense(large_image, strength=0.5)
        self.assertIsInstance(result, Image.Image)

        # Test with grayscale image
        gray_image = Image.new('L', (100, 100), color=128)
        result = smart_crop(gray_image, aspect_ratio="1:1")
        self.assertIsInstance(result, Image.Image)

    def test_parameter_validation(self):
        """Test parameter validation in protection methods."""
        # Test invalid intensity values (should use defaults)
        result = apply_cloaking(self.test_image, intensity=-0.1)
        self.assertIsInstance(result, Image.Image)
        
        result = apply_cloaking(self.test_image, intensity=1.5)
        self.assertIsInstance(result, Image.Image)

        # Test invalid strength values
        result = apply_style_defense(self.test_image, strength=-0.1)
        self.assertIsInstance(result, Image.Image)
        
        result = apply_style_defense(self.test_image, strength=1.5)
        self.assertIsInstance(result, Image.Image)


class TestImageFormats(unittest.TestCase):
    """Test different image formats compatibility."""

    def test_jpeg_format(self):
        """Test JPEG format handling."""
        # Create and process JPEG image
        jpeg_path = 'test_temp.jpg'
        image = Image.new('RGB', (100, 100), color='red')
        image.save(jpeg_path, 'JPEG')
        
        # Test loading and processing
        with Image.open(jpeg_path) as loaded:
            result = apply_cloaking(loaded)
        self.assertIsInstance(result, Image.Image)
        
        # Clean up
        if os.path.exists(jpeg_path):
            os.remove(jpeg_path)

    def test_png_format(self):
        """Test PNG format handling."""
        # Create and process PNG image
        png_path = 'test_temp.png'
        image = Image.new('RGBA', (100, 100), color=(255, 0, 0, 255))
        image.save(png_path, 'PNG')
        
        # Test loading and processing
        with Image.open(png_path) as loaded:
            result = apply_style_defense(loaded)
        self.assertIsInstance(result, Image.Image)
        
        # Clean up
        if os.path.exists(png_path):
            os.remove(png_path)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)