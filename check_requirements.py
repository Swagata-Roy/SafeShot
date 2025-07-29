#!/usr/bin/env python3
"""
SafeShot - Requirements Checker
Checks for missing dependencies and provides helpful installation instructions.
"""

import sys
import subprocess
import importlib.util
from typing import List, Tuple, Optional

def check_module(module_name: str, pip_name: Optional[str] = None) -> Tuple[bool, str]:
    """Check if a module is available and return status message."""
    if pip_name is None:
        pip_name = module_name
    
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return False, f"❌ {module_name} - Not found"
        else:
            # Try to import to ensure it's properly installed
            __import__(module_name)
            return True, f"✅ {module_name} - OK"
    except ImportError as e:
        return False, f"❌ {module_name} - Import error: {e}"

ModuleRequirement = Tuple[str, bool, Optional[str]]

def check_system_requirements() -> List[Tuple[str, bool, str]]:
    """Check all required and optional dependencies."""
    requirements: List[ModuleRequirement] = [
        # Core requirements
        ("gradio", True, None),
        ("PIL", True, "pillow"),
        ("cv2", True, "opencv-python"),
        ("numpy", True, None),
        ("scipy", True, None),
        ("skimage", True, "scikit-image"),
        ("matplotlib", True, None),
        ("torch", True, None),
        ("torchvision", True, None),
        ("face_recognition", True, None),
        ("tqdm", True, None),
        ("requests", True, None),
    ]
    
    results: List[Tuple[str, bool, str]] = []
    for module, required, pip_name in requirements:
        _, message = check_module(module, pip_name or module)  # Ignore success flag since we check message content
        results.append((module, required, message))
    
    return results

def install_missing_packages(missing_packages: List[str]) -> bool:
    """Attempt to install missing packages using pip."""
    if not missing_packages:
        return True
    
    print("\n📦 Installing missing packages...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install"
        ] + missing_packages)
        print("✅ Installation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False

def main():
    """Main requirements checking function."""
    print("🔍 SafeShot - Requirements Checker")
    print("=" * 40)
    
    results = check_system_requirements()
    
    missing_packages: List[str] = []
    critical_missing: List[str] = []
    
    print("\n📋 Checking dependencies:")
    for module, required, message in results:
        print(f"  {message}")
        if "❌" in message:
            pip_name = module if module != "cv2" else "opencv-python"
            pip_name = "pillow" if module == "PIL" else pip_name
            pip_name = "scikit-image" if module == "skimage" else pip_name
            
            missing_packages.append(pip_name)
            if required:
                critical_missing.append(pip_name)
    
    if not missing_packages:
        print("\n🎉 All dependencies are satisfied!")
        print("\nYou can now run SafeShot with:")
        print("  python app.py")
        return True
    
    print(f"\n⚠️  Found {len(missing_packages)} missing package(s)")
    
    if critical_missing:
        print(f"🚨 Critical packages missing: {', '.join(critical_missing)}")
    
    # Offer to install missing packages
    if missing_packages:
        response = input("\nWould you like to install missing packages automatically? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            success = install_missing_packages(missing_packages)
            if success:
                print("\n🔄 Re-checking requirements...")
                results = check_system_requirements()
                missing_after: List[str] = [r[0] for r in results if "❌" in r[2]]
                if not missing_after:
                    print("✅ All packages installed successfully!")
                    return True
                else:
                    print("❌ Some packages still missing after installation")
    
    # Manual installation instructions
    print("\n📖 Manual installation:")
    if missing_packages:
        print("Run the following command:")
        print(f"  pip install {' '.join(missing_packages)}")
    
    print("\nOr install all requirements at once:")
    print("  pip install -r requirements.txt")
    
    return len(critical_missing) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)