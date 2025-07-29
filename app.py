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
    metadata
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
    watermark_image: ImageType
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
    watermark_image: ImageType
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
                crop_enabled,
                crop_ratio,
                edge_softness,
                crop_sensitivity,
                strip_exif,
                add_watermark,
                watermark_text,
                watermark_opacity,
                watermark_image
            )

            if protected_image:
                # Save protected image to a buffer
                img_buffer = io.BytesIO()
                protected_image.save(img_buffer, format="PNG")
                img_buffer.seek(0)
                zip_file.writestr(f"protected_image_{i+1}.png", img_buffer.read())

    zip_buffer.seek(0)
    # Save to a temporary file and return the path
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
        tmp_file.write(zip_buffer.getvalue())
        tmp_path = tmp_file.name
    return tmp_path, "✅ Batch processing complete. Download the zip file."


def create_interface():
    """Create and configure the Gradio interface."""
    
    with gr.Blocks(title="SafeShot - Image Protection Tool") as app:
        gr.Markdown(
            """
            # 🛡️ SafeShot - Image Protection Tool
            
            Protect your images from unauthorized AI training and misuse with multiple defense mechanisms.
            """
        )
        
        with gr.Tabs():
            with gr.TabItem("Single Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Input section
                        input_image = gr.Image(
                            label="Upload Image",
                            type="pil"
                        )
                        
                        protection_method = gr.Radio(
                            choices=["Cloaking (Anti-AI)", "Style Defense", "None (Metadata Only)"],
                            value="Cloaking (Anti-AI)",
                            label="Primary Protection Method"
                        )
                        
                        # Cloaking options
                        with gr.Group(visible=True) as cloak_options:
                            gr.Markdown("### Cloaking Options")
                            cloak_intensity = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.5,
                                step=0.1,
                                label="Cloaking Intensity"
                            )
                            cloak_method = gr.Dropdown(
                                choices=["fawkes", "lowkey"],
                                value="fawkes",
                                label="Cloaking Method"
                            )
                        
                        # Style defense options
                        with gr.Group(visible=False) as style_options:
                            gr.Markdown("### Style Defense Options")
                            style_strength = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.5,
                                step=0.1,
                                label="Defense Strength"
                            )
                            texture_type = gr.Dropdown(
                                choices=["noise", "blur", "pixelate", "swirl"],
                                value="noise",
                                label="Texture Type"
                            )
                        
                        # Cropping options
                        with gr.Group():
                            gr.Markdown("### Cropping Options")
                            crop_enabled = gr.Checkbox(
                                label="Enable Smart Cropping",
                                value=False
                            )
                            crop_ratio = gr.Dropdown(
                                choices=["1:1", "4:3", "16:9", "9:16", "Original"],
                                value="Original",
                                label="Aspect Ratio"
                            )
                            edge_softness = gr.Slider(
                                minimum=0,
                                maximum=50,
                                value=10,
                                step=5,
                                label="Edge Softness (pixels)"
                            )
                            crop_sensitivity = gr.Dropdown(
                                choices=["center", "face", "saliency"],
                                value="face",
                                label="Crop Focus (Sensitivity)"
                            )
                        
                        # Metadata options
                        with gr.Group():
                            gr.Markdown("### Metadata Options")
                            strip_exif = gr.Checkbox(
                                label="Strip EXIF Data",
                                value=True
                            )
                            add_watermark = gr.Checkbox(
                                label="Add Watermark",
                                value=False
                            )
                            watermark_text = gr.Textbox(
                                label="Watermark Text",
                                placeholder="© Your Name",
                                visible=False
                            )
                            watermark_image = gr.Image(
                                label="Watermark Image",
                                type="pil",
                                visible=False
                            )
                            watermark_opacity = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.3,
                                step=0.1,
                                label="Watermark Opacity",
                                visible=False
                            )
                        
                        protect_btn = gr.Button(
                            "🛡️ Protect Image",
                            variant="primary",
                            size="lg"
                        )
                        
                    with gr.Column(scale=1):
                        # Output section
                        output_image = gr.Image(
                            label="Protected Image",
                            type="pil"
                        )
                        status_text = gr.Textbox(
                            label="Status",
                            interactive=False
                        )
                        
                        with gr.Row():
                            reset_btn = gr.Button(
                                "🔄 Reset",
                                variant="secondary"
                            )
                
                # Examples with explicit type annotation
                examples: List[List[str]] = [
                    ["assets/example1.png"],
                    ["assets/example2.png"],
                    ["assets/example3.png"]
                ]
                gr.Examples(
                    examples=examples,
                    inputs=input_image,
                    label="Example Images"
                )
                
                # Event handlers
                def update_options_visibility(method: str) -> Dict[gr.Group, GradioUpdateType]:
                    return {
                        cloak_options: gr.update(visible=(method == "Cloaking (Anti-AI)")),
                        style_options: gr.update(visible=(method == "Style Defense"))
                    }
                
                def update_watermark_visibility(enabled: bool) -> Dict[gr.Component, GradioUpdateType]:
                    return {
                        watermark_text: gr.update(visible=enabled),
                        watermark_image: gr.update(visible=enabled),
                        watermark_opacity: gr.update(visible=enabled)
                    }
                
                protection_method.change(
                    update_options_visibility,
                    inputs=[protection_method],
                    outputs=[cloak_options, style_options]
                )
                
                add_watermark.change(
                    update_watermark_visibility,
                    inputs=[add_watermark],
                    outputs=[watermark_text, watermark_image, watermark_opacity]
                )
                
                protect_btn.click(
                    protect_image_single,
                    inputs=[
                        input_image,
                        protection_method,
                        cloak_intensity,
                        cloak_method,
                        style_strength,
                        texture_type,
                        crop_enabled,
                        crop_ratio,
                        edge_softness,
                        crop_sensitivity,
                        strip_exif,
                        add_watermark,
                        watermark_text,
                        watermark_opacity,
                        watermark_image
                    ],
                    outputs=[output_image, status_text]
                )
                
                # Reset function with proper type hints
                def reset_fn() -> Tuple[None, None, str]:
                    return None, None, "Ready to protect a new image."

                reset_btn.click(
                    reset_fn,
                    outputs=[input_image, output_image, status_text]
                )

            with gr.TabItem("Batch Processing"):
                with gr.Row():
                    with gr.Column(scale=1):
                        batch_input_images = gr.File(
                            label="Upload Images",
                            file_count="multiple",
                            file_types=["image"]
                        )
                        
                        batch_protection_method = gr.Radio(
                            choices=["Cloaking (Anti-AI)", "Style Defense", "None (Metadata Only)"],
                            value="Cloaking (Anti-AI)",
                            label="Primary Protection Method"
                        )
                        
                        with gr.Group(visible=True) as batch_cloak_options:
                            gr.Markdown("### Cloaking Options")
                            batch_cloak_intensity = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.5,
                                step=0.1,
                                label="Cloaking Intensity"
                            )
                            batch_cloak_method = gr.Dropdown(
                                choices=["Fawkes-style", "LowKey-style"],
                                value="Fawkes-style",
                                label="Cloaking Method"
                            )
                        
                        with gr.Group(visible=False) as batch_style_options:
                            gr.Markdown("### Style Defense Options")
                            batch_style_strength = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.5,
                                step=0.1,
                                label="Defense Strength"
                            )
                            batch_texture_type = gr.Dropdown(
                                choices=["noise", "blur", "pixelate", "swirl"],
                                value="noise",
                                label="Texture Type"
                            )
                        
                        with gr.Group():
                            gr.Markdown("### Cropping Options")
                            batch_crop_enabled = gr.Checkbox(
                                label="Enable Smart Cropping",
                                value=False
                            )
                            batch_crop_ratio = gr.Dropdown(
                                choices=["1:1", "4:3", "16:9", "9:16", "Original"],
                                value="Original",
                                label="Aspect Ratio"
                            )
                            batch_edge_softness = gr.Slider(
                                minimum=0,
                                maximum=50,
                                value=10,
                                step=5,
                                label="Edge Softness (pixels)"
                            )
                            batch_crop_sensitivity = gr.Dropdown(
                                choices=["center", "face", "saliency"],
                                value="face",
                                label="Crop Focus (Sensitivity)"
                            )
                        
                        with gr.Group():
                            gr.Markdown("### Metadata Options")
                            batch_strip_exif = gr.Checkbox(
                                label="Strip EXIF Data",
                                value=True
                            )
                            batch_add_watermark = gr.Checkbox(
                                label="Add Watermark",
                                value=False
                            )
                            batch_watermark_text = gr.Textbox(
                                label="Watermark Text",
                                placeholder="© Your Name",
                                visible=False
                            )
                            batch_watermark_image = gr.Image(
                                label="Watermark Image",
                                type="pil",
                                visible=False
                            )
                            batch_watermark_opacity = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.3,
                                step=0.1,
                                label="Watermark Opacity",
                                visible=False
                            )
                        
                        batch_protect_btn = gr.Button(
                            "🛡️ Protect All Images",
                            variant="primary",
                            size="lg"
                        )
                        
                    with gr.Column(scale=1):
                        batch_output_file = gr.File(label="Download Protected Images (Zip)")
                        batch_status_text = gr.Textbox(
                            label="Status",
                            interactive=False
                        )

                def batch_update_options_visibility(method: str) -> Dict[gr.Group, GradioUpdateType]:
                    return {
                        batch_cloak_options: gr.update(visible=(method == "Cloaking (Anti-AI)")),
                        batch_style_options: gr.update(visible=(method == "Style Defense"))
                    }
                
                def batch_update_watermark_visibility(enabled: bool) -> Dict[gr.Component, GradioUpdateType]:
                    return {
                        batch_watermark_text: gr.update(visible=enabled),
                        batch_watermark_image: gr.update(visible=enabled),
                        batch_watermark_opacity: gr.update(visible=enabled)
                    }
                
                batch_protection_method.change(
                    batch_update_options_visibility,
                    inputs=[batch_protection_method],
                    outputs=[batch_cloak_options, batch_style_options]
                )
                
                batch_add_watermark.change(
                    batch_update_watermark_visibility,
                    inputs=[batch_add_watermark],
                    outputs=[batch_watermark_text, batch_watermark_image, batch_watermark_opacity]
                )

                batch_protect_btn.click(
                    protect_images_batch,
                    inputs=[
                        batch_input_images,
                        batch_protection_method,
                        batch_cloak_intensity,
                        batch_cloak_method,
                        batch_style_strength,
                        batch_texture_type,
                        batch_crop_enabled,
                        batch_crop_ratio,
                        batch_edge_softness,
                        batch_crop_sensitivity,
                        batch_strip_exif,
                        batch_add_watermark,
                        batch_watermark_text,
                        batch_watermark_opacity,
                        batch_watermark_image
                    ],
                    outputs=[batch_output_file, batch_status_text]
                )
        
    return app


if __name__ == "__main__":
    app = create_interface()
    app.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7861,  # Use a different port
        show_error=True
    )