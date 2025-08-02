# 🛡️ SafeShot - Image Protection Tool

SafeShot is a comprehensive image protection tool designed to safeguard your photos from unauthorized AI training and misuse. It provides multiple defense mechanisms including AI cloaking, style defense, smart cropping, and metadata protection.

## 🚀 Features

### Protection Methods
- **AI Cloaking (Anti-AI)**: Adds imperceptible adversarial noise to prevent AI models from learning from your images
- **Style Defense**: Applies texture warping and blending techniques to disrupt style transfer and AI analysis
- **FaceShield (Anti-FaceSwap)**: Disrupts facial recognition models to protect against face swapping and deepfakes.
- **Smart Cropping**: Intelligently crops images with soft edges and enhanced face detection to remove identifying features
- **Metadata Protection**: Strips EXIF data more robustly and adds customizable watermarks

### Key Capabilities
- **Multiple Protection Layers**: Combine different methods for enhanced security
- **Batch Processing**: Protect multiple images at once
- **Quality Preservation**: Maintain image quality while applying protections
- **Easy-to-use Interface**: Simple Gradio web interface
- **Cross-platform**: Works on Windows, macOS, and Linux

## 📦 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Quick Start
```bash
# Clone or download the project
cd SafeShot

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

The application will open in your browser at `http://localhost:7860`

## 🎯 Usage

### Basic Usage
1. **Upload Image**: Drag and drop or click to upload your image
2. **Select Protection Method**: Choose from Cloaking, Style Defense, or FaceShield
3. **Configure Options**: Adjust protection intensity and other parameters
4. **Apply Protection**: Click "Protect Image" to apply selected protections
5. **Download**: Save the protected image to your device

### Advanced Options

#### Cloaking (Anti-AI)
- **Intensity**: Controls the strength of adversarial noise (0.1-1.0)
- **Method**: Choose between Fawkes-style and LowKey-style cloaking

#### Style Defense
- **Strength**: Controls texture warping intensity (0.1-1.0)
- **Texture Type**: Select from subtle, moderate, or aggressive texture patterns

#### FaceShield (Anti-FaceSwap)
- **Perturbation Intensity**: Controls the strength of the adversarial attack on facial embeddings.
- **Defense Method**: Choose between `attention_manipulation` and `embedding_disruption`.
- **Imperceptibility Blur**: Applies a Gaussian blur to make the perturbations less visible.

#### Smart Cropping
- **Aspect Ratio**: Maintain or change image proportions
- **Edge Softness**: Apply feathering to crop edges
- **Sensitivity**: Control how aggressively to crop

#### Metadata Protection
- **Strip EXIF**: Remove all metadata including GPS, camera info, etc.
- **Add Watermark**: Overlay text or image watermarks
- **Watermark Customization**: Adjust opacity, position, and text

## 🔧 Technical Details

### External Libraries
- **MediaPipe**: Used for advanced face and landmark detection, enhancing smart cropping capabilities.
- **ImageIO**: Provides extended support for various image formats, improving compatibility.
- **Piexif**: Enables robust handling and stripping of EXIF metadata for enhanced privacy.

### Architecture
SafeShot/
├── app.py                      # Main Gradio interface
├── image_protection/           # Core protection modules
│   ├── __init__.py             # Initializes module
│   ├── cloak.py                # AI cloaking algorithms
│   ├── cropper.py              # Smart cropping with edge softening
│   ├── metadata.py             # EXIF scrubbing & watermarking
│   ├── style_defense.py        # Texture warping & blending
│   ├── faceshield.py           # Face-swapping and deepfake defense
│   ├── utils.py                # Shared image utilities
│   └── version.py              # Version metadata
├── assets/                     # Logo & example images
│   ├── logo.png                # SafeShot branding icon
│   ├── example1.png            # Image samples
│   ├── example2.png
│   └── example3.png
├── tests/                      # Unit and feature tests
│   ├── __init__.py
│   └── test_basic.py           # Functional coverage
├── check_requirements.py       # Dependency sanity checker
├── .env                        # Environment variables
├── .gitattributes              # Git metadata control
├── LICENSE                     # Apache-2.0 license
├── requirements.txt            # Dependency manifest
└── README.md                   # Project documentation

### Protection Algorithms

#### AI Cloaking
- **Fawkes-style**: Adds targeted perturbations to facial features
- **LowKey-style**: Applies frequency domain perturbations

#### Style Defense
- **Texture Warping**: Distorts local texture patterns
- **Color Manipulation**: Subtle color space transformations
- **Edge Enhancement**: Modifies edge characteristics

#### Smart Cropping
- **Saliency Detection**: Identifies important image regions
- **Aspect Ratio Optimization**: Maintains visual balance
- **Edge Feathering**: Creates smooth transitions

## 🛠️ Development

### Project Structure
- **app.py**: Main application entry point with Gradio interface
- **image_protection/**: Modular protection system
- **assets/**: Example images and test data

### Adding New Protection Methods
1. Create new module in `image_protection/`
2. Implement protection function with standard interface
3. Add UI controls in `app.py`
4. Update documentation

### Testing
```bash
# Run basic tests
python -m pytest tests/

# Test individual modules
python -c "from image_protection import cloak; print('Cloaking module loaded')"
```

## 📊 Performance

### Processing Speed
- **Small images** (≤1MP): ~1-2 seconds
- **Medium images** (1-5MP): ~3-5 seconds
- **Large images** (5-20MP): ~5-15 seconds

### Memory Usage
- **Base application**: ~200MB
- **Per image**: ~50-200MB depending on size

### Quality Impact
- **Cloaking**: <1% perceptual difference
- **Style Defense**: <3% perceptual difference
- **Smart Cropping**: Variable based on crop ratio

## 🔒 Privacy & Security

### Data Handling
- **Local Processing**: All processing happens on your device
- **No Uploads**: Images never leave your computer
- **No Tracking**: No analytics or data collection
- **Temporary Files**: All temporary files are cleaned up

### Protection Effectiveness
- **AI Training**: Significantly reduces AI model accuracy
- **Style Transfer**: Disrupts style extraction algorithms
- **Face Recognition**: Reduces facial recognition accuracy
- **Reverse Engineering**: Makes reconstruction difficult

## 🐛 Troubleshooting

### Common Issues

#### "Module not found" errors
```bash
pip install -r requirements.txt --upgrade
```

#### "OpenCV not found" errors
```bash
pip install opencv-python
```

#### Memory issues with large images
- Reduce image size before processing
- Close other applications
- Increase system RAM if possible

#### Slow processing
- Use smaller images for testing
- Reduce protection intensity
- Close unnecessary browser tabs

### Getting Help
- Check the [Issues](https://github.com/your-repo/SafeShot/issues) page
- Review error logs in the terminal
- Try with example images first

## 🤝 Contributing

Contributions welcome!

### Ways to Contribute
- Report bugs and issues
- Suggest new protection methods
- Improve documentation
- Add new features
- Test with different image types

## 📄 License

This project is licensed under the Apache-2.0 license - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Fawkes Project**: For inspiration on AI cloaking techniques
- **LowKey**: For adversarial perturbation methods
- **Gradio**: For the excellent web interface framework
- **OpenCV**: For computer vision capabilities
- **PIL/Pillow**: For image processing support

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/your-repo/SafeShot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/SafeShot/discussions)
---

**⚠️ Disclaimer**: While SafeShot provides strong protection against many AI systems, no protection method is 100% effective. Use multiple protection methods for best results.