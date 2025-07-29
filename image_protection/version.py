"""
SafeShot - Image Protection Tool
Version information and module metadata.
"""
from typing import List, Dict, Union

__version__ = "1.0.0"
__author__ = "Swagata Roy"
__description__ = "Comprehensive image protection against AI training and misuse"
__license__ = "Apache-2.0"
__url__ = "https://github.com/Swagata-Roy/SafeShot"

# Version info tuple for easy comparison
VERSION_INFO = (1, 0, 0)

# Protection methods available
PROTECTION_METHODS = [
    "cloak",
    "style_defense", 
    "cropper",
    "metadata"
]

# Configuration type definitions
ConfigValue = Union[str, bool, float]
ConfigDict = Dict[str, ConfigValue]

# Default configuration
DEFAULT_CONFIG: Dict[str, ConfigDict] = {
    "cloak": {
        "intensity": 0.5,
        "method": "fawkes"
    },
    "style_defense": {
        "strength": 0.3,
        "texture_type": "subtle"
    },
    "cropper": {
        "aspect_ratio": "original",
        "edge_softness": 0.2,
        "sensitivity": 0.5
    },
    "metadata": {
        "strip_exif": True,
        "add_watermark": False,
        "watermark_text": "Protected by SafeShot"
    }
}

def get_version() -> str:
    """Return the current version as a string."""
    return __version__

def get_version_info() -> tuple[int, int, int]:
    """Return version as a tuple for comparison."""
    return VERSION_INFO

def get_protection_methods() -> List[str]:
    """Return list of available protection methods."""
    return PROTECTION_METHODS.copy()

def get_default_config() -> Dict[str, ConfigDict]:
    """Return default configuration dictionary."""
    return DEFAULT_CONFIG.copy()