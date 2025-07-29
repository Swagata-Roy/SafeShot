"""
SafeShot Image Protection Module

This module provides various image protection techniques to prevent
unauthorized AI training and misuse of personal images.
"""

from . import cloak
from . import style_defense
from . import cropper
from . import metadata
from . import utils

__version__ = "1.0.0"
__all__ = ["cloak", "style_defense", "cropper", "metadata", "utils"]