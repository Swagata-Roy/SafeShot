from typing import Optional, Tuple, Union, Dict, Any, List
import gradio as gr
import tempfile
import numpy as np
from PIL import Image
import numpy.typing as npt
import zipfile
import io

# Import our protection modules
from image_protection import (
    cloak,
    style_defense,
    cropper,
    metadata,
    faceshield
)

# Type aliases
ImageType = Union[Image.Image, npt.NDArray[np.uint8], None]
GradioUpdateType = Dict[str, Any]


def protect_image_single(
    image: ImageType,
    protection_method: str,
    # Cloaking options
    cloak_intensity: float,
    cloak_method: str,
    # Style defense options
    style_strength: float,
    texture_type: str,
    # FaceShield options
    faceshield_intensity: float,
    faceshield_method: str,
    faceshield_blur: float,
    # Cropping options
    crop_enabled: bool,
    crop_ratio: str,
    edge_softness: int,
    crop_sensitivity: str,
    # Metadata options
    strip_exif: bool,
    add_watermark: bool,
    watermark_text: str,
    watermark_opacity: float,
    watermark_image: ImageType,
    # Invisible watermark options
    add_invisible_watermark: bool,
    invisible_watermark_text: str
) -> Tuple[Optional[Image.Image], str]:
    """
    Main function to apply selected protection methods to a single image.
    """
    if image is None:
        return None, "Please upload an image first."
    
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    protected_image = image.copy()
    applied_methods: List[str] = []
    
    try:
        # Apply protection methods based on selection
        if protection_method == "Cloaking (Anti-AI)":
            protected_image = cloak.apply_cloaking(
                protected_image, 
                intensity=cloak_intensity,
                method=cloak_method
            )
            applied_methods.append(f"AI Cloaking ({cloak_method})")
            
        elif protection_method == "Style Defense":
            protected_image = style_defense.apply_style_defense(
                protected_image,
                strength=style_strength,
                texture_type=texture_type
            )
            applied_methods.append("Style Defense")
            
        elif protection_method == "FaceShield (Anti-FaceSwap)":
            protected_image = faceshield.apply_protection(
                protected_image,
                faceshield_intensity=faceshield_intensity,
                faceshield_method=faceshield_method,
                faceshield_blur=faceshield_blur
            )
            applied_methods.append(f"FaceShield ({faceshield_method})")

        # Apply cropping if enabled
        if crop_enabled:
            protected_image = cropper.smart_crop(
                protected_image,
                aspect_ratio=crop_ratio,
                edge_softness=edge_softness,
                focus_mode=crop_sensitivity
            )
            applied_methods.append("Smart Cropping")
        
        # Apply metadata protection
        if strip_exif:
            protected_image = metadata.strip_metadata(protected_image)
            applied_methods.append("EXIF Stripped")
            
        if add_watermark:
            if watermark_text:
                protected_image = metadata.add_watermark(
                    protected_image,
                    text=watermark_text,
                    opacity=watermark_opacity
                )
                applied_methods.append("Text Watermark Added")
            if watermark_image is not None:
                if isinstance(watermark_image, np.ndarray):
                    watermark_image = Image.fromarray(watermark_image)
                protected_image = metadata.add_image_watermark(
                    protected_image,
                    watermark_image=watermark_image,
                    opacity=watermark_opacity
                )
                applied_methods.append("Image Watermark Added")
        
        if add_invisible_watermark and invisible_watermark_text:
            protected_image = metadata.add_invisible_watermark(
                protected_image,
                signature=invisible_watermark_text
            )
            applied_methods.append("Invisible Watermark Added")
            
        status = f"✅ Protection applied: {', '.join(applied_methods)}"
        return protected_image, status
        
    except Exception as e:
        return image, f"❌ Error: {str(e)}"


def protect_images_batch(
    images: List[ImageType],
    protection_method: str,
    # Cloaking options
    cloak_intensity: float,
    cloak_method: str,
    # Style defense options
    style_strength: float,
    texture_type: str,
    # FaceShield options
    faceshield_intensity: float,
    faceshield_method: str,
    faceshield_blur: float,
    # Cropping options
    crop_enabled: bool,
    crop_ratio: str,
    edge_softness: int,
    crop_sensitivity: str,
    # Metadata options
    strip_exif: bool,
    add_watermark: bool,
    watermark_text: str,
    watermark_opacity: float,
    watermark_image: ImageType,
    # Invisible watermark options
    add_invisible_watermark: bool,
    invisible_watermark_text: str
) -> Tuple[Optional[str], str]:
    """
    Main function to apply selected protection methods to a batch of images.
    """
    if not images:
        return None, "Please upload one or more images."

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for i, image in enumerate(images):
            if image is None:
                continue
            
            # Open image if it's a string (path), otherwise use it directly
            img = Image.open(image) if isinstance(image, str) else image
            protected_image, _ = protect_image_single(
                img,
                protection_method,
                cloak_intensity,
                cloak_method,
                style_strength,
                texture_type,
                faceshield_intensity,
                faceshield_method,
                faceshield_blur,
                crop_enabled,
                crop_ratio,
                edge_softness,
                crop_sensitivity,
                strip_exif,
                add_watermark,
                watermark_text,
                watermark_opacity,
                watermark_image,
                add_invisible_watermark,
                invisible_watermark_text
            )

            if protected_image:
                # Save protected image to a buffer
                img_buffer = io.BytesIO()
                protected_image.save(img_buffer, format="PNG")
                img_buffer.seek(0)
                zip_file.writestr(f"protected_image_{i+1}.png", img_buffer.read())

    zip_buffer.seek(0)
    zip_content = zip_buffer.getvalue()

    # Save to a temporary file and return the path
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
        tmp_file.write(zip_content)
        tmp_path = tmp_file.name

    # Ensure the file is properly closed before returning
    zip_buffer.close()

    return tmp_path, "✅ Batch processing complete. Download the zip file."


def create_interface():
    """Create and configure the Gradio interface."""
    
    with gr.Blocks(title="SafeShot - Image Protection Tool") as app:
        gr.HTML(
            """
            <div style="display: flex; align-items: center;">
                <img src="/gradio_api/file=assets/logo.png" width="50" style="margin-right: 10px;"/>
                <h1 style="margin: 0;">SafeShot - Image Protection Tool</h1>
            </div>
            <p>Protect your images from unauthorized AI training and misuse with multiple defense mechanisms.</p>
            """
        )
        
        with gr.Tabs():
            with gr.TabItem("Single Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Input section
                        input_image = gr.Image(label="Upload Image", type="pil")
                        
                        protection_method = gr.Radio(
                            choices=["Cloaking (Anti-AI)", "Style Defense", "FaceShield (Anti-FaceSwap)", "None (Metadata Only)"],
                            value="Cloaking (Anti-AI)",
                            label="Primary Protection Method"
                        )
                        
                        # --- Options Groups ---
                        with gr.Group(visible=True) as cloak_options:
                            gr.Markdown("### Cloaking Options")
                            cloak_intensity = gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.1, label="Cloaking Intensity")
                            cloak_method = gr.Dropdown(choices=["fawkes", "lowkey"], value="fawkes", label="Cloaking Method")
                        
                        with gr.Group(visible=False) as style_options:
                            gr.Markdown("### Style Defense Options")
                            style_strength = gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.1, label="Defense Strength")
                            texture_type = gr.Dropdown(choices=["noise", "blur", "pixelate", "swirl"], value="noise", label="Texture Type")
                        
                        with gr.Group(visible=False) as faceshield_options:
                            gr.Markdown("### FaceShield Options")
                            faceshield_intensity = gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.1, label="Perturbation Intensity")
                            faceshield_method = gr.Dropdown(choices=["attention_manipulation", "embedding_disruption"], value="attention_manipulation", label="Defense Method")
                            faceshield_blur = gr.Slider(minimum=0.0, maximum=5.0, value=1.0, step=0.5, label="Imperceptibility Blur")

                        with gr.Group():
                            gr.Markdown("### Cropping Options")
                            crop_enabled = gr.Checkbox(label="Enable Smart Cropping", value=False)
                            crop_ratio = gr.Dropdown(choices=["1:1", "4:3", "16:9", "9:16", "Original"], value="Original", label="Aspect Ratio")
                            edge_softness = gr.Slider(minimum=0, maximum=50, value=10, step=5, label="Edge Softness (pixels)")
                            crop_sensitivity = gr.Dropdown(choices=["center", "face", "saliency"], value="face", label="Crop Focus (Sensitivity)")
                        
                        with gr.Group():
                            gr.Markdown("### Metadata Options")
                            strip_exif = gr.Checkbox(label="Strip EXIF Data", value=True)
                            add_watermark = gr.Checkbox(label="Add Watermark", value=False)
                            watermark_text = gr.Textbox(label="Watermark Text", placeholder="© Your Name", visible=False)
                            watermark_image = gr.Image(label="Watermark Image", type="pil", visible=False)
                            watermark_opacity = gr.Slider(minimum=0.1, maximum=1.0, value=0.3, step=0.1, label="Watermark Opacity", visible=False)
                            
                            gr.Markdown("---")
                            add_invisible_watermark = gr.Checkbox(label="Add Invisible Watermark (LSB)", value=False)
                            invisible_watermark_text = gr.Textbox(label="Invisible Watermark Text", placeholder="Enter a secret message...", visible=False)

                        protect_btn = gr.Button("🛡️ Protect Image", variant="primary", size="lg")
                        
                    with gr.Column(scale=1):
                        # Output section
                        output_image = gr.Image(label="Protected Image", type="pil", format="png")
                        status_text = gr.Textbox(label="Status", interactive=False)
                        with gr.Row():
                            reset_btn = gr.Button("🔄 Reset", variant="secondary")
                
                examples: List[List[str]] = [["assets/example1.png"], ["assets/example2.png"], ["assets/example3.png"]]
                gr.Examples(examples=examples, inputs=input_image, label="Example Images")
                
                # --- Event Handlers (Single Image) ---
                def update_options_visibility(method: str) -> Dict[gr.Group, GradioUpdateType]:
                    return {
                        cloak_options: gr.update(visible=(method == "Cloaking (Anti-AI)")),
                        style_options: gr.update(visible=(method == "Style Defense")),
                        faceshield_options: gr.update(visible=(method == "FaceShield (Anti-FaceSwap)"))
                    }
                
                def update_watermark_visibility(enabled: bool) -> Dict[gr.Component, GradioUpdateType]:
                    return {
                        watermark_text: gr.update(visible=enabled),
                        watermark_image: gr.update(visible=enabled),
                        watermark_opacity: gr.update(visible=enabled)
                    }
                
                def update_invisible_watermark_visibility(enabled: bool) -> Dict[gr.Component, GradioUpdateType]:
                    return {
                        invisible_watermark_text: gr.update(visible=enabled)
                    }
                
                protection_method.change(update_options_visibility, inputs=[protection_method], outputs=[cloak_options, style_options, faceshield_options])
                add_watermark.change(update_watermark_visibility, inputs=[add_watermark], outputs=[watermark_text, watermark_image, watermark_opacity])
                add_invisible_watermark.change(update_invisible_watermark_visibility, inputs=[add_invisible_watermark], outputs=[invisible_watermark_text])

                protect_btn.click(
                    protect_image_single,
                    inputs=[
                        input_image, protection_method,
                        cloak_intensity, cloak_method,
                        style_strength, texture_type,
                        faceshield_intensity, faceshield_method, faceshield_blur,
                        crop_enabled, crop_ratio, edge_softness, crop_sensitivity,
                        strip_exif, add_watermark, watermark_text, watermark_opacity, watermark_image,
                        add_invisible_watermark, invisible_watermark_text
                    ],
                    outputs=[output_image, status_text]
                )
                
                def reset_fn() -> Tuple[None, None, str]:
                    return None, None, "Ready to protect a new image."
                reset_btn.click(reset_fn, outputs=[input_image, output_image, status_text])

            with gr.TabItem("Batch Processing"):
                with gr.Row():
                    with gr.Column(scale=1):
                        batch_input_images = gr.File(label="Upload Images", file_count="multiple", file_types=["image"])
                        
                        batch_protection_method = gr.Radio(
                            choices=["Cloaking (Anti-AI)", "Style Defense", "FaceShield (Anti-FaceSwap)", "None (Metadata Only)"],
                            value="Cloaking (Anti-AI)",
                            label="Primary Protection Method"
                        )
                        
                        # --- Options Groups (Batch) ---
                        with gr.Group(visible=True) as batch_cloak_options:
                            gr.Markdown("### Cloaking Options")
                            batch_cloak_intensity = gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.1, label="Cloaking Intensity")
                            batch_cloak_method = gr.Dropdown(choices=["Fawkes-style", "LowKey-style"], value="Fawkes-style", label="Cloaking Method")
                        
                        with gr.Group(visible=False) as batch_style_options:
                            gr.Markdown("### Style Defense Options")
                            batch_style_strength = gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.1, label="Defense Strength")
                            batch_texture_type = gr.Dropdown(choices=["noise", "blur", "pixelate", "swirl"], value="noise", label="Texture Type")
                        
                        with gr.Group(visible=False) as batch_faceshield_options:
                            gr.Markdown("### FaceShield Options")
                            batch_faceshield_intensity = gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.1, label="Perturbation Intensity")
                            batch_faceshield_method = gr.Dropdown(choices=["attention_manipulation", "embedding_disruption"], value="attention_manipulation", label="Defense Method")
                            batch_faceshield_blur = gr.Slider(minimum=0.0, maximum=5.0, value=1.0, step=0.5, label="Imperceptibility Blur")

                        with gr.Group():
                            gr.Markdown("### Cropping Options")
                            batch_crop_enabled = gr.Checkbox(label="Enable Smart Cropping", value=False)
                            batch_crop_ratio = gr.Dropdown(choices=["1:1", "4:3", "16:9", "9:16", "Original"], value="Original", label="Aspect Ratio")
                            batch_edge_softness = gr.Slider(minimum=0, maximum=50, value=10, step=5, label="Edge Softness (pixels)")
                            batch_crop_sensitivity = gr.Dropdown(choices=["center", "face", "saliency"], value="face", label="Crop Focus (Sensitivity)")
                        
                        with gr.Group():
                            gr.Markdown("### Metadata Options")
                            batch_strip_exif = gr.Checkbox(label="Strip EXIF Data", value=True)
                            batch_add_watermark = gr.Checkbox(label="Add Watermark", value=False)
                            batch_watermark_text = gr.Textbox(label="Watermark Text", placeholder="© Your Name", visible=False)
                            batch_watermark_image = gr.Image(label="Watermark Image", type="pil", visible=False)
                            batch_watermark_opacity = gr.Slider(minimum=0.1, maximum=1.0, value=0.3, step=0.1, label="Watermark Opacity", visible=False)
                            
                            gr.Markdown("---")
                            batch_add_invisible_watermark = gr.Checkbox(label="Add Invisible Watermark (LSB)", value=False)
                            batch_invisible_watermark_text = gr.Textbox(label="Invisible Watermark Text", placeholder="Enter a secret message...", visible=False)

                        batch_protect_btn = gr.Button("🛡️ Protect All Images", variant="primary", size="lg")
                        
                    with gr.Column(scale=1):
                        batch_output_file = gr.File(label="Download Protected Images (Zip)")
                        batch_status_text = gr.Textbox(label="Status", interactive=False)

                # --- Event Handlers (Batch) ---
                def batch_update_options_visibility(method: str) -> Dict[gr.Group, GradioUpdateType]:
                    return {
                        batch_cloak_options: gr.update(visible=(method == "Cloaking (Anti-AI)")),
                        batch_style_options: gr.update(visible=(method == "Style Defense")),
                        batch_faceshield_options: gr.update(visible=(method == "FaceShield (Anti-FaceSwap)"))
                    }
                
                def batch_update_watermark_visibility(enabled: bool) -> Dict[gr.Component, GradioUpdateType]:
                    return {
                        batch_watermark_text: gr.update(visible=enabled),
                        batch_watermark_image: gr.update(visible=enabled),
                        batch_watermark_opacity: gr.update(visible=enabled)
                    }
                
                def batch_update_invisible_watermark_visibility(enabled: bool) -> Dict[gr.Component, GradioUpdateType]:
                    return {
                        batch_invisible_watermark_text: gr.update(visible=enabled)
                    }
                
                batch_protection_method.change(batch_update_options_visibility, inputs=[batch_protection_method], outputs=[batch_cloak_options, batch_style_options, batch_faceshield_options])
                batch_add_watermark.change(batch_update_watermark_visibility, inputs=[batch_add_watermark], outputs=[batch_watermark_text, batch_watermark_image, batch_watermark_opacity])
                batch_add_invisible_watermark.change(batch_update_invisible_watermark_visibility, inputs=[batch_add_invisible_watermark], outputs=[batch_invisible_watermark_text])

                batch_protect_btn.click(
                    protect_images_batch,
                    inputs=[
                        batch_input_images, batch_protection_method,
                        batch_cloak_intensity, batch_cloak_method,
                        batch_style_strength, batch_texture_type,
                        batch_faceshield_intensity, batch_faceshield_method, batch_faceshield_blur,
                        batch_crop_enabled, batch_crop_ratio, batch_edge_softness, batch_crop_sensitivity,
                        batch_strip_exif, batch_add_watermark, batch_watermark_text, batch_watermark_opacity, batch_watermark_image,
                        batch_add_invisible_watermark, batch_invisible_watermark_text
                    ],
                    outputs=[batch_output_file, batch_status_text]
                )

            with gr.TabItem("Extract Watermark"):
                with gr.Row():
                    with gr.Column():
                        extract_image_input = gr.Image(label="Upload Image to Extract Watermark", type="pil")
                        extract_button = gr.Button("🔍 Extract Invisible Watermark", variant="primary")
                    with gr.Column():
                        extracted_text_output = gr.Textbox(label="Extracted Watermark", interactive=False)
                        extraction_status = gr.Textbox(label="Status", interactive=False)

                def extract_watermark_fn(image: ImageType) -> Tuple[str, str]:
                    if image is None:
                        return "", "Please upload an image."
                    try:
                        if isinstance(image, np.ndarray):
                            image = Image.fromarray(image)
                        
                        watermark = metadata.extract_invisible_watermark(image)
                        
                        if watermark:
                            return watermark, "✅ Watermark extracted successfully."
                        else:
                            return "", "⚠️ No invisible watermark found."
                    except Exception as e:
                        return "", f"❌ Error during extraction: {str(e)}"

                extract_button.click(
                    extract_watermark_fn,
                    inputs=[extract_image_input],
                    outputs=[extracted_text_output, extraction_status]
                )

            with gr.TabItem("Guide"):
                gr.HTML(
                    """
                    <div style="max-width: 800px; margin: 0 auto; padding: 20px;">
                        <h1 style="text-align: center; color: #2c3e50;">📖 SafeShot User Guide</h1>
                        
                        <h2 style="color: #3498db;">🎯 What is SafeShot?</h2>
                        <p>SafeShot is a comprehensive image protection tool designed to safeguard your photos from unauthorized AI training and misuse. It offers multiple defense mechanisms to protect your visual content while maintaining image quality.</p>
                        
                        <h2 style="color: #3498db;">🔧 Protection Methods</h2>
                        
                        <h3 style="color: #e74c3c;">1. AI Cloaking (Anti-AI)</h3>
                        <p><strong>What it does:</strong> Applies subtle pixel-level perturbations that are invisible to humans but confuse AI models, preventing them from learning from your images.</p>
                        <ul>
                            <li><strong>Fawkes-style:</strong> Advanced cloaking technique that creates "poisoned" images for facial recognition systems</li>
                            <li><strong>LowKey-style:</strong> More aggressive cloaking that works against broader AI training datasets</li>
                        </ul>
                        <p><strong>Best for:</strong> Photos with faces, personal images, professional headshots</p>
                        
                        <h3 style="color: #e74c3c;">2. Style Defense</h3>
                        <p><strong>What it does:</strong> Applies artistic transformations that maintain visual appeal while disrupting AI feature extraction.</p>
                        <ul>
                            <li><strong>Noise:</strong> Adds controlled noise patterns</li>
                            <li><strong>Blur:</strong> Applies selective blurring to key areas</li>
                            <li><strong>Pixelate:</strong> Creates pixelation effects</li>
                            <li><strong>Swirl:</strong> Applies swirling distortions</li>
                        </ul>
                        <p><strong>Best for:</strong> Artistic content, social media posts, creative portfolios</p>
                        
                        <h3 style="color: #e74c3c;">3. FaceShield (Anti-FaceSwap)</h3>
                        <p><strong>What it does:</strong> Disrupts facial recognition models by applying adversarial perturbations to image embeddings. This makes it difficult for models like InsightFace to extract reliable facial features, thus protecting against face swapping and deepfakes.</p>
                        <ul>
                            <li><strong>Attention Manipulation:</strong> Adds noise to salient (important) regions of the face to confuse the model's focus.</li>
                            <li><strong>Embedding Disruption:</strong> Uses adversarial attacks (like PGD) to subtly alter the image in a way that corrupts its feature embedding.</li>
                        </ul>
                        <p><strong>Best for:</strong> Protecting profile pictures, headshots, and any images where facial identity is important.</p>

                        <h2 style="color: #3498db;">✂️ Smart Cropping</h2>
                        <p><strong>What it does:</strong> Intelligently crops images to remove sensitive areas while preserving important content.</p>
                        <ul>
                            <li><strong>Aspect Ratios:</strong> 1:1, 4:3, 16:9, 9:16, or keep original</li>
                            <li><strong>Focus Modes:</strong>
                                <ul>
                                    <li><strong>Center:</strong> Crops from the center</li>
                                    <li><strong>Face:</strong> Prioritizes face detection</li>
                                    <li><strong>Saliency:</strong> Uses AI to identify important regions</li>
                                </ul>
                            </li>
                            <li><strong>Edge Softness:</strong> Smooths crop edges for natural blending</li>
                        </ul>
                        
                        <h2 style="color: #3498db;">🛡️ Metadata Protection</h2>
                        <ul>
                            <li><strong>Strip EXIF:</strong> Removes location, device, and timestamp data</li>
                            <li><strong>Text Watermark:</strong> Adds custom text overlays</li>
                            <li><strong>Image Watermark:</strong> Embeds logo or signature images</li>
                            <li><strong>Opacity Control:</strong> Adjust watermark transparency</li>
                            <li><strong>Invisible Watermark:</strong> Embed a hidden message (signature) into the image using LSB steganography. This is useful for proving ownership later.</li>
                        </ul>
                        
                        <h2 style="color: #3498db;">📋 How to Use</h2>
                        
                        <h3 style="color: #27ae60;">Single Image Processing</h3>
                        <ol>
                            <li>Go to the <strong>"Single Image"</strong> tab</li>
                            <li>Upload your image using the upload button</li>
                            <li>Select your preferred protection method:
                                <ul>
                                    <li>Choose <strong>"Cloaking (Anti-AI)"</strong> for AI protection</li>
                                    <li>Choose <strong>"Style Defense"</strong> for artistic protection</li>
                                    <li>Choose <strong>"None (Metadata Only)"</strong> for just metadata/watermark changes</li>
                                </ul>
                            </li>
                            <li>Adjust method-specific settings (intensity, texture type, etc.)</li>
                            <li>Configure optional features:
                                <ul>
                                    <li>Enable <strong>Smart Cropping</strong> to resize/crop</li>
                                    <li>Enable <strong>Strip EXIF</strong> to remove metadata</li>
                                    <li>Add <strong>watermarks</strong> for branding/copyright</li>
                                    <li>Enable <strong>Invisible Watermark</strong> and enter a secret message to embed it in the image.</li>
                                </ul>
                            </li>
                            <li>Click <strong>"🛡️ Protect Image"</strong></li>
                            <li>Download your protected image</li>
                        </ol>
                        
                        <h3 style="color: #27ae60;">Batch Processing</h3>
                        <ol>
                            <li>Go to the <strong>"Batch Processing"</strong> tab</li>
                            <li>Upload multiple images (drag & drop or click to select)</li>
                            <li>Configure protection settings (same as single image)</li>
                            <li>Click <strong>"🛡️ Protect All Images"</strong></li>
                            <li>Download the ZIP file containing all protected images</li>
                        </ol>
                        
                        <h3 style="color: #27ae60;">Extracting an Invisible Watermark</h3>
                        <ol>
                            <li>Go to the <strong>"Extract Watermark"</strong> tab</li>
                            <li>Upload an image that you suspect contains a hidden watermark</li>
                            <li>Click <strong>"🔍 Extract Invisible Watermark"</strong></li>
                            <li>If a watermark is found, it will be displayed in the text box.</li>
                        </ol>
                        
                        <h2 style="color: #3498db;">💡 Pro Tips</h2>
                        <ul>
                            <li><strong>Test different methods:</strong> Try various protection levels to find the best balance</li>
                            <li><strong>Batch similar images:</strong> Group similar images for consistent batch processing</li>
                            <li><strong>Keep originals:</strong> Always save unprotected copies as backup</li>
                            <li><strong>Check results:</strong> Preview protected images before sharing</li>
                        </ul>
                        
                        <h2 style="color: #3498db;">🚨 Important Notes</h2>
                        <ul>
                            <li>Protected images may look slightly different - this is normal and intentional</li>
                            <li>AI cloaking effectiveness varies by image type and AI model</li>
                            <li>Style defense creates visible changes - choose based on your needs</li>
                            <li>Metadata stripping is permanent - save originals first</li>
                        </ul>
                    </div>
                    """
                )
        
        gr.Markdown(
            """
            <p align="center">Made with ❤️ by <a href="https://github.com/Swagata-Roy/SafeShot" target="_blank" style="text-decoration: none; color: #555;"><strong>Swagata Roy</strong></a></p>
            """
        )
        
    return app


if __name__ == "__main__":
    app = create_interface()
    app.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7861,  # Use a different port
        show_error=True,
        allowed_paths=["assets"]
    )