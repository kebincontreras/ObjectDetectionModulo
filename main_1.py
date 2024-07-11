import gradio as gr
from image_processing import apply_blur, clip_image, wrap_image
from detection import *
from detection import yolov10_inference, calculate_detection_metrics, yolov10_inference_1, save_detections
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
    
    image = Image.open(image)
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


    url_1 = "C:\\Users\\USUARIO\\Documents\\GitHub\\Yolov10\\kitti"

    original_annotated, original_detections = yolov10_inference(original_image, model_id, image_size, conf_threshold)
    clipped_annotated, clipped_detections = yolov10_inference((clipped_image*255.0).astype(np.uint8), model_id, image_size, conf_threshold)
    wrapped_annotated, wrapped_detections = yolov10_inference(wrapped_image, model_id, image_size, conf_threshold)

    # Assuming `recons` is a function in `utils.py`
    recon_image = recons(img_tensor, DO=1, L=1.0, vertical=(vertical == "True"), t=t)
    recon_image_pil = transforms.ToPILImage()(recon_image.squeeze(0))
    recon_image_np = np.array(recon_image_pil).astype(np.uint8)


    recon_annotated, recon_detections = yolov10_inference(recon_image_np, model_id, image_size, conf_threshold)


    yolov10_inference_1(original_image, model_id, image_size, conf_threshold, url_1)
    yolov10_inference_1(clipped_image, model_id, image_size, conf_threshold, url_1)
    yolov10_inference_1(wrapped_image, model_id, image_size, conf_threshold, url_1)

    metrics_clip = calculate_detection_metrics(original_detections, clipped_detections)
    metrics_wrap = calculate_detection_metrics(original_detections, wrapped_detections)
    metrics_recons = calculate_detection_metrics(original_detections, recon_detections)

    return original_annotated, clipped_annotated, wrapped_annotated, recon_annotated, metrics_clip, metrics_wrap, metrics_recons

if __name__ == "__main__":
    # Ejemplo de uso
    url = "C:\\Users\\USUARIO\\Documents\\GitHub\\Yolov10\\kitti\\image_2\\000008.png"
    model_id = "yolov10m"
    image_size = 640
    conf_threshold = 0.20
    correction = 1
    sat_factor =1.5
    kernel_size = 7
    DO = "1"
    t = 0.6
    vertical = "True"
    result = process_image(url, model_id, image_size, conf_threshold, correction, sat_factor, kernel_size, DO, t, vertical) 
    # Desempaquetar los resultados
    original_annotated, clipped_annotated, wrapped_annotated, recon_annotated, metrics_clip, metrics_wrap, metrics_recons = result


    # Mostrar imágenes
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # Cambia el tamaño según sea necesario
    axs[0].imshow(original_annotated)
    axs[0].set_title('Original Annotated')
    axs[0].axis('off')

    axs[1].imshow(clipped_annotated)
    axs[1].set_title('Clipped Annotated')
    axs[1].axis('off')

    axs[2].imshow(wrapped_annotated)
    axs[2].set_title('Wrapped Annotated')
    axs[2].axis('off')

    axs[3].imshow(recon_annotated)
    axs[3].set_title('Reconstructed Annotated')
    axs[3].axis('off')

    plt.show()

    # Imprimir métricas
    print("Metrics for Clipped Image:", metrics_clip)
    print("Metrics for Wrapped Image:", metrics_wrap)
    print("Metrics for Reconstructed Image:", metrics_recons)







