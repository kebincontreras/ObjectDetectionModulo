
import gradio as gr
from image_processing import apply_blur, clip_image, wrap_image
from detection import yolov10_inference, calculate_detection_metrics
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch
from utils import *
#from utils import flip_odd_lines, modulo, center_modulo, unmodulo, hard_thresholding, stripe_estimation, recons

def process_image(image, model_id, image_size, conf_threshold, correction, sat_factor, kernel_size, DO, t, vertical):

    original_image = np.array(image)
    original_image = original_image - original_image.min()
    original_image = original_image / original_image.max()
    original_image = original_image * 255.0
    original_image = original_image.astype(np.uint8)


    blurred_image = apply_blur(original_image, kernel_size)
    clipped_image = clip_image(blurred_image / 255.0, correction, sat_factor) 
    wrapped_image = wrap_image(blurred_image / 255.0, correction, sat_factor) 

    original_annotated, original_detections = yolov10_inference(original_image, model_id, image_size, conf_threshold)
    clipped_annotated, clipped_detections = yolov10_inference((clipped_image*255.0).astype(np.uint8), model_id, image_size, conf_threshold)
    wrapped_annotated, wrapped_detections = yolov10_inference((wrapped_image*255.0).astype(np.uint8), model_id, image_size, conf_threshold)

    # img_tensor = transforms.ToTensor()(wrapped_image).unsqueeze(0)
    img_tensor = torch.tensor(wrapped_image, dtype=torch.float32 ).permute(2, 0, 1).unsqueeze(0)
    # Assuming `recons` is a function in `utils.py`
    recon_image = recons(img_tensor, DO=1, L=1.0, vertical=(vertical == "True"), t=t)
    recon_image_pil = transforms.ToPILImage()(recon_image.squeeze(0))
    recon_image_np = np.array(recon_image_pil) * 255.0
    recon_annotated, recon_detections = yolov10_inference(recon_image_np.astype(np.uint8), model_id, image_size, conf_threshold)

    metrics_clip = calculate_detection_metrics(original_detections, clipped_detections)
    metrics_wrap = calculate_detection_metrics(original_detections, wrapped_detections)
    metrics_recons = calculate_detection_metrics(original_detections, recon_detections)

    return original_annotated, clipped_annotated, wrapped_annotated, recon_annotated, metrics_clip, metrics_wrap, metrics_recons

def app():
    with gr.Blocks() as demo:
        gr.Markdown("### YOLOv10 Object Detection on Original and Modified Images")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Upload Image")
                model_id = gr.Dropdown(label="Model", choices=["yolov10n", "yolov10s", "yolov10m", "yolov10b", "yolov10l", "yolov10x"], value="yolov10m")
                image_size = gr.Slider(label="Image Size", minimum=320, maximum=1280, step=32, value=640)
                conf_threshold = gr.Slider(label="Confidence Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.85)
                correction = gr.Slider(label="Correction Factor", minimum=0, maximum=1.0, step=0.1, value=0.4)
                sat_factor = gr.Slider(label="Saturation Factor", minimum=1.0, maximum=5.0, step=0.1, value=3.0)
                kernel_size = gr.Slider(label="Blur Kernel Size", minimum=1, maximum=7, step=1, value=1)
                DO = gr.Radio(label="DO", choices=["1", "2"], value="1")
                t = gr.Slider(label="t", minimum=0.0, maximum=1.0, step=0.1, value=0.5)
                vertical = gr.Radio(label="Vertical", choices=["True", "False"], value="False")
                process_button = gr.Button("Process Image")

            with gr.Column():
                output_original = gr.Image(label="Original Image with Annotations")
                output_clip = gr.Image(label="Clipped Image with Annotations")
                output_wrap = gr.Image(label="Wrapped Image with Annotations")
                output_recons = gr.Image(label="Reconstructed Image with Annotations")
                metrics_clip = gr.Textbox(label="Metrics for Clipped Image")
                metrics_wrap = gr.Textbox(label="Metrics for Wrapped Image")
                metrics_recons = gr.Textbox(label="Metrics for Reconstructed Image")

        process_button.click(
            fn=process_image,
            inputs=[image, model_id, image_size, conf_threshold, correction, sat_factor, kernel_size, DO, t, vertical],
            outputs=[output_original, output_clip, output_wrap, output_recons, metrics_clip, metrics_wrap, metrics_recons]
        )

    return demo

if __name__ == "__main__":
    app().launch()
