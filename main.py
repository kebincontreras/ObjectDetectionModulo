
import gradio as gr
from image_processing import apply_blur, clip_image, wrap_image
from detection import yolov10_inference, calculate_detection_metrics
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch
from utils import *
#from utils import flip_odd_lines, modulo, center_modulo, unmodulo, hard_thresholding, stripe_estimation, recons
from utils import modulo
import cv2

import matplotlib.pyplot as plt


def process_image(image, model_id, image_size, conf_threshold, correction, sat_factor, kernel_size, DO, t, vertical):

    original_image = np.array(image)
    original_image = original_image - original_image.min()
    original_image = original_image / original_image.max()
    original_image = original_image * 255.0
    original_image = original_image.astype(np.uint8)

    # scaling factor
    scaling = 1.0
    original_image = cv2.resize(original_image, (0, 0), fx=scaling, fy=scaling)


    blurred_image = apply_blur(original_image / 255.0, kernel_size)
    clipped_image = clip_image(blurred_image, correction, sat_factor) 

    img_tensor = torch.tensor(blurred_image, dtype=torch.float32 ).permute(2, 0, 1).unsqueeze(0)
    img_tensor = modulo( img_tensor * sat_factor, L=1.0)

    wrapped_image = img_tensor.squeeze(0).permute(1, 2, 0).numpy()
    wrapped_image = (wrapped_image*255).astype(np.uint8)

    original_annotated, original_detections = yolov10_inference(original_image, model_id, image_size, conf_threshold)
    clipped_annotated, clipped_detections = yolov10_inference((clipped_image*255.0).astype(np.uint8), model_id, image_size, conf_threshold)
    wrapped_annotated, wrapped_detections = yolov10_inference(wrapped_image, model_id, image_size, conf_threshold)

    # Assuming `recons` is a function in `utils.py`
    recon_image = recons(img_tensor, DO=1, L=1.0, vertical=(vertical == "True"), t=t)
    recon_image_pil = transforms.ToPILImage()(recon_image.squeeze(0))
    recon_image_np = np.array(recon_image_pil).astype(np.uint8)


    recon_annotated, recon_detections = yolov10_inference(recon_image_np, model_id, image_size, conf_threshold)

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
                model_id = gr.Dropdown(label="Model", choices=["yolov10n", "yolov10s", "yolov10m", "yolov10b", "yolov10l", "yolov10x"], value="yolov10x")
                image_size = gr.Slider(label="Image Size", minimum=320, maximum=1280, step=32, value=640)
                conf_threshold = gr.Slider(label="Confidence Threshold", minimum=0.0, maximum=1.0, step=0.01, value=0.85)
                correction = gr.Slider(label="Correction Factor", minimum=0, maximum=1.0, step=0.1, value=1.0)
                sat_factor = gr.Slider(label="Saturation Factor", minimum=1.0, maximum=5.0, step=0.1, value=2.0)
                kernel_size = gr.Slider(label="Blur Kernel Size", minimum=1, maximum=7, step=1, value=7)
                DO = gr.Radio(label="DO", choices=["1", "2"], value="1")
                t = gr.Slider(label="t", minimum=0.0, maximum=1.0, step=0.1, value=0.5)
                vertical = gr.Radio(label="Vertical", choices=["True", "False"], value="True")
                process_button = gr.Button("Process Image")

            with gr.Column():
                output_original = gr.Image(label="Original + blur")
                output_clip = gr.Image(label="Clipped ")
                output_wrap = gr.Image(label="Wrapped")
                output_recons = gr.Image(label="Reconstructed")
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
