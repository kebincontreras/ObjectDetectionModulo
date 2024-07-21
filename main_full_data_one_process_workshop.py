import os
import gradio as gr
from image_processing import apply_blur, clip_image, wrap_image, save_images
from detection import *
from detection import yolov10_inference, calculate_detection_metrics, save_detections, read_kitti_annotations, yolov10_inference_1
from notacion import map_yolo_classes_to_db
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch
from utils import *
from utils import modulo
import cv2
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_global_metrics(metrics_list):
    metrics_sum = {}
    for metrics in metrics_list:
        for key, value in metrics.items():
            if key not in metrics_sum:
                metrics_sum[key] = 0
            metrics_sum[key] += value
    
    metrics_count = len(metrics_list)
    mean_metrics = {key: value / metrics_count for key, value in metrics_sum.items()}
    
    return mean_metrics

def process_image(image_path, annotations_path, model_id, image_size, conf_threshold, correction, sat_factor, kernel_size, DO, t, vertical):
    image = Image.open(image_path)
    original_image = np.array(image)

    original_image = original_image - original_image.min()
    original_image = original_image / original_image.max()
    original_image = original_image * 255.0
    original_image = original_image.astype(np.uint8)

    scaling = 1.0
    original_image = cv2.resize(original_image, (0, 0), fx=scaling, fy=scaling)

    blurred_image = apply_blur(original_image / 255.0, kernel_size)
    clipped_image = clip_image(blurred_image, correction, sat_factor) 

    img_tensor = torch.tensor(blurred_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    img_tensor = modulo(img_tensor * sat_factor, L=1.0)

    wrapped_image = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    wrapped_image = (wrapped_image * 255).astype(np.uint8)

    original_annotated, LDR_detections = yolov10_inference_1(original_image, model_id, image_size, conf_threshold)
    clipped_annotated, clipped_detections = yolov10_inference_1((clipped_image * 255.0).astype(np.uint8), model_id, image_size, conf_threshold)
    wrapped_annotated, wrapped_detections = yolov10_inference_1(wrapped_image, model_id, image_size, conf_threshold)

    recon_image = recons(img_tensor, DO=1, L=1.0, vertical=(vertical == "True"), t=t)
    
    recon_image_pil = transforms.ToPILImage()(recon_image.squeeze(0).cpu())
    recon_image_np = np.array(recon_image_pil).astype(np.uint8)
    recon_annotated, recon_detections = yolov10_inference_1(recon_image_np, model_id, image_size, conf_threshold)

    original_annotations = read_kitti_annotations(annotations_path)

    LDR_detections = map_yolo_classes_to_db(LDR_detections)
    clipped_detections = map_yolo_classes_to_db(clipped_detections)
    wrapped_detections = map_yolo_classes_to_db(wrapped_detections)
    recon_detections = map_yolo_classes_to_db(recon_detections)

    return original_annotations, LDR_detections, clipped_detections, wrapped_detections, recon_detections

def process_dataset(dataset_dir, model_id, image_size, conf_threshold, correction, sat_factor, kernel_size, DO, t, vertical, ranges, process_type):
    image_dir = os.path.join(dataset_dir, 'image')
    label_dir = os.path.join(dataset_dir, 'label')

    all_metrics = []

    for image_name in os.listdir(image_dir):
        if image_name.endswith('.png'):
            image_id = image_name.split('.')[0]
            # Check if image_id falls within any of the specified ranges
            if any(start <= image_id <= end for start, end in ranges):
                image_path = os.path.join(image_dir, image_name)
                label_path = os.path.join(label_dir, image_name.replace('.png', '.txt'))

                original_annotations, LDR_detections, clipped_detections, wrapped_detections, recon_detections = process_image(
                    image_path, label_path, model_id, image_size, conf_threshold, correction, sat_factor, kernel_size, DO, t, vertical
                )

                if process_type == "original":
                    metrics = calculate_detection_metrics_1(original_annotations, LDR_detections)
                elif process_type == "clip":
                    metrics = calculate_detection_metrics_1(original_annotations, clipped_detections)
                elif process_type == "wrap":
                    metrics = calculate_detection_metrics_1(original_annotations, wrapped_detections)
                elif process_type == "recon":
                    metrics = calculate_detection_metrics_1(original_annotations, recon_detections)
                else:
                    raise ValueError("Invalid process_type. Choose from 'original', 'clip', 'wrap', 'recon'.")

                all_metrics.append(metrics)

    global_metrics = calculate_global_metrics(all_metrics) if all_metrics else {}

    save_metrics_to_txt(dataset_dir, image_size, conf_threshold, correction, sat_factor, kernel_size, DO, t, vertical, global_metrics, process_type, ranges)

    return global_metrics


def save_metrics_to_txt(dataset_dir, image_size, conf_threshold, correction, sat_factor, kernel_size, DO, t, vertical, metrics, process_type, ranges):
    range_str = "_".join([f"{start}_{end}" for start, end in ranges])
    metrics_file_name = f"sat_{sat_factor}_t_{t}_threshold_{conf_threshold}_kernel_{kernel_size}_{process_type}_metrics_{range_str}_workshop.txt"
    metrics_file_path = os.path.join(dataset_dir, metrics_file_name)
    with open(metrics_file_path, 'w') as f:
        f.write("Configuration Parameters:\n")
        f.write(f"image_size: {image_size}\n")
        f.write(f"conf_threshold: {conf_threshold}\n")
        f.write(f"correction: {correction}\n")
        f.write(f"sat_factor: {sat_factor}\n")
        f.write(f"kernel_size: {kernel_size}\n")
        f.write(f"DO: {DO}\n")
        f.write(f"t: {t}\n")
        f.write(f"vertical: {vertical}\n")
        f.write(f"\nImage Ranges Processed: {', '.join([f'{start}-{end}' for start, end in ranges])}\n")
        f.write("\nGlobal Metrics:\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    print(f"Global metrics and configuration parameters saved to {metrics_file_path}")


if __name__ == "__main__":
    dataset_dir = "C:\\Users\\USUARIO\\Documents\\GitHub\\Yolov10\\kitti"
    model_id = "yolov10x"
    image_size = 640
    conf_threshold = 0.60
    correction = 1
    sat_factor = 2
    kernel_size = 7
    DO = "1"
    t = 0.6
    vertical = "True"

    # Define the ranges of image IDs to process
    ranges = [('000715', '000815')]

    # Specify which process to execute: 'original', 'clip', 'wrap', or 'recon'
    process_type = 'recon'  # Change this to the desired process type

    global_metrics = process_dataset(
        dataset_dir, model_id, image_size, conf_threshold, correction, sat_factor, kernel_size, DO, t, vertical, ranges, process_type
    )

    # Print global metrics
    print(f"Global Metrics for {process_type.capitalize()} Images:", global_metrics)
