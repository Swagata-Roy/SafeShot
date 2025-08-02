"""
Advanced tests for SafeShot image protection functionality.

This module contains a comprehensive suite of tests for all features,
including edge cases and parameter variations.
"""

import unittest
from PIL import Image
import os
import sys
import numpy as np
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_protection.cloak import apply_cloaking, validate_cloaking_effectiveness
from image_protection.style_defense import apply_style_defense
from image_protection.cropper import create_vignette, adaptive_crop_for_privacy
from image_protection.metadata import (
    get_metadata_info,
    add_image_watermark,
    add_invisible_watermark,
    extract_invisible_watermark,
    add_copyright_banner,
    add_timestamp_watermark,
    apply_pattern_watermark,
)
from image_protection.faceshield import apply_faceshield, apply_adversarial_embedding
from image_protection.utils import detect_faces
from image_protection.version import get_version, get_protection_methods


class TestSafeShotAdvanced(unittest.TestCase):
    """Advanced test suite for SafeShot image protection functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_image = Image.new('RGB', (200, 150), color='blue')
        self.test_image_path = 'test_advanced_temp.png'
        self.test_image.save(self.test_image_path)

        # Use a real image for face detection tests
        self.face_image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'example1.png')
        self.face_image = Image.open(self.face_image_path)

        self.watermark_image = Image.new('RGBA', (50, 20), color=(255, 0, 0, 128))
        self.watermark_image_path = 'test_watermark_temp.png'
        self.watermark_image.save(self.watermark_image_path)

    def tearDown(self):
        """Clean up test fixtures."""
        for path in [self.test_image_path, self.watermark_image_path]:
            if os.path.exists(path):
                os.remove(path)

    # Cloaking Tests
    def test_cloaking_methods(self):
        """Test different cloaking methods."""
        for method in ["fawkes", "lowkey"]:
            with self.subTest(method=method):
                cloaked = apply_cloaking(self.face_image, intensity=0.5, method=method)
                self.assertIsInstance(cloaked, Image.Image)
                self.assertEqual(cloaked.size, self.face_image.size)
                # Check that the image has been modified
                self.assertFalse(np.array_equal(np.array(self.face_image), np.array(cloaked)))

    def test_cloaking_effectiveness_validation(self):
        """Test the validation of cloaking effectiveness."""
        cloaked = apply_cloaking(self.test_image, intensity=0.1)
        metrics: Dict[str, Any] = validate_cloaking_effectiveness(self.test_image, cloaked)
        self.assertIsInstance(metrics, dict)
        self.assertIn("psnr", metrics)
        self.assertIn("ssim", metrics)
        self.assertIn("avg_perturbation", metrics)
        self.assertIn("quality_assessment", metrics)
        self.assertGreater(metrics.get("psnr", 0), 20)
        self.assertGreater(metrics.get("ssim", 0), 0.8)

    # Cropper Tests
    def test_vignette_creation(self):
        """Test vignette creation."""
        vignetted = create_vignette(self.test_image, strength=0.7)
        self.assertIsInstance(vignetted, Image.Image)
        self.assertEqual(vignetted.size, self.test_image.size)
        # Check that the corners are darker
        corner_pixel = vignetted.getpixel((0, 0))
        center_pixel = self.test_image.getpixel((100, 75))
        if isinstance(corner_pixel, int) and isinstance(center_pixel, int):
            self.assertLess(corner_pixel, center_pixel)
        elif isinstance(corner_pixel, tuple) and isinstance(center_pixel, tuple):
            self.assertLess(sum(corner_pixel), sum(center_pixel))

    def test_adaptive_crop(self):
        """Test adaptive privacy cropping."""
        cropped = adaptive_crop_for_privacy(self.face_image)
        self.assertIsInstance(cropped, Image.Image)
        self.assertTrue(cropped.size[0] <= self.face_image.size[0])
        self.assertTrue(cropped.size[1] <= self.face_image.size[1])

    # FaceShield Tests
    def test_faceshield_methods(self):
        """Test different faceshield methods."""
        for method in ["embedding_disruption", "attention_manipulation"]:
            with self.subTest(method=method):
                protected = apply_faceshield(self.face_image, intensity=0.5, method=method)
                self.assertIsInstance(protected, Image.Image)
                self.assertEqual(protected.size, self.face_image.size)

    def test_adversarial_embedding(self):
        """Test adversarial embedding attack."""
        for method in ["pgd", "fgsm"]:
            with self.subTest(method=method):
                perturbed = apply_adversarial_embedding(self.face_image, attack_method=method, epsilon=0.05)
                self.assertIsInstance(perturbed, Image.Image)
                self.assertEqual(perturbed.size, self.face_image.size)
                self.assertFalse(np.array_equal(np.array(self.face_image), np.array(perturbed)))

    # Metadata Tests
    def test_get_metadata(self):
        """Test metadata extraction."""
        # This test is limited as we create a basic image, but we can check the structure
        with Image.open(self.test_image_path) as img:
            metadata = get_metadata_info(img)
        self.assertIsInstance(metadata, dict)
        self.assertIn('exif', metadata)
        self.assertIn('info', metadata)
        self.assertEqual(metadata['format'], 'PNG')

    def test_image_watermark(self):
        """Test adding an image as a watermark."""
        watermarked = add_image_watermark(self.test_image, self.watermark_image, scale=0.2)
        self.assertIsInstance(watermarked, Image.Image)
        self.assertEqual(watermarked.size, self.test_image.size)

    def test_invisible_watermark(self):
        """Test invisible watermark embedding and extraction."""
        signature = "SafeShotTest"
        watermarked = add_invisible_watermark(self.test_image, signature)
        extracted_signature = extract_invisible_watermark(watermarked)
        self.assertEqual(signature, extracted_signature)

    def test_copyright_banner(self):
        """Test adding a copyright banner."""
        bannered = add_copyright_banner(self.test_image, "© 2024 SafeShot")
        self.assertIsInstance(bannered, Image.Image)
        self.assertEqual(bannered.width, self.test_image.width)
        self.assertGreater(bannered.height, self.test_image.height)

    def test_timestamp_watermark(self):
        """Test adding a timestamp watermark."""
        timestamped = add_timestamp_watermark(self.test_image)
        self.assertIsInstance(timestamped, Image.Image)

    def test_pattern_watermark(self):
        """Test applying a repeating pattern watermark."""
        patterned = apply_pattern_watermark(self.test_image, "PROTECTED")
        self.assertIsInstance(patterned, Image.Image)
        self.assertEqual(patterned.size, self.test_image.size)

    # Style Defense Tests
    def test_style_defense_methods(self):
        """Test different style defense methods."""
        for texture_type in ["noise", "blur", "pixelate", "swirl"]:
            with self.subTest(texture_type=texture_type):
                defended = apply_style_defense(self.test_image, strength=0.5, texture_type=texture_type)
                self.assertIsInstance(defended, Image.Image)
                self.assertEqual(defended.size, self.test_image.size)
                self.assertFalse(np.array_equal(np.array(self.test_image), np.array(defended)))

    # Utils Tests
    def test_face_detection(self):
        """Test face detection utility."""
        faces = detect_faces(self.face_image)
        self.assertIsInstance(faces, list)
        self.assertGreater(len(faces), 0)
        self.assertEqual(len(faces[0]), 4) # (x, y, w, h)

    # Versioning Tests
    def test_version_and_methods(self):
        """Test version and protection methods retrieval."""
        version = get_version()
        methods = get_protection_methods()
        self.assertIsInstance(version, str)
        self.assertIsInstance(methods, list)
        self.assertIn("faceshield", methods)

if __name__ == '__main__':
    unittest.main(verbosity=2)