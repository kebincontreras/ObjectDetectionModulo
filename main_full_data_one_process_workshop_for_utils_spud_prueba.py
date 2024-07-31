import os
import time
from PIL import Image
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from image_processing import apply_blur, clip_image, wrap_image
from detection import yolov10_inference_1, calculate_detection_metrics_1, read_kitti_annotations
from notacion import map_yolo_classes_to_db
from utils import modulo
from utils_spud import recons_spud

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
    start_time = time.time()

    image = Image.open(image_path)
    original_image = np.array(image)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    original_image = original_image - original_image.min()
    original_image = original_image / original_image.max() * 255.0
    original_image = original_image.astype(np.uint8)
    
    blurred_image = apply_blur(original_image, kernel_size)
    clipped_image = clip_image(blurred_image, correction, sat_factor)

    img_tensor = torch.tensor(blurred_image / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    img_tensor = modulo(img_tensor * sat_factor, L=1.0)

    wrapped_image = (img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    original_start = time.time()
    original_annotated, LDR_detections = yolov10_inference_1(original_image, model_id, image_size, conf_threshold)
    original_time = time.time() - original_start

    clipped_start = time.time()
    clipped_annotated, clipped_detections = yolov10_inference_1(clipped_image, model_id, image_size, conf_threshold)
    clipped_time = time.time() - clipped_start

    wrapped_start = time.time()
    wrapped_annotated, wrapped_detections = yolov10_inference_1(wrapped_image, model_id, image_size, conf_threshold)
    wrapped_time = time.time() - wrapped_start

    recon_start = time.time()
    recon_image = recons_spud(img_tensor, threshold=0.1, mx=1.0)
    recon_image_pil = transforms.ToPILImage()(recon_image.squeeze(0).cpu())
    recon_image_np = np.array(recon_image_pil).astype(np.uint8)
    recon_annotated, recon_detections = yolov10_inference_1(recon_image_np, model_id, image_size, conf_threshold)
    recon_time = time.time() - recon_start

    total_time = time.time() - start_time

    original_annotations = read_kitti_annotations(annotations_path)
    LDR_detections = map_yolo_classes_to_db(LDR_detections)
    clipped_detections = map_yolo_classes_to_db(clipped_detections)
    wrapped_detections = map_yolo_classes_to_db(wrapped_detections)
    recon_detections = map_yolo_classes_to_db(recon_detections)

    times = {'original_time': original_time, 'clipped_time': clipped_time, 'wrapped_time': wrapped_time, 'recon_time': recon_time, 'total_time': total_time}

    return original_annotations, LDR_detections, clipped_detections, wrapped_detections, recon_detections, times

def process_dataset(dataset_dir, model_ids, image_size, conf_threshold, correction, sat_factors, kernel_size, DO, t, vertical, ranges, process_types):
    image_dir = os.path.join(dataset_dir, 'image')
    label_dir = os.path.join(dataset_dir, 'label')
    all_metrics = {(process_type, sat_factor, model_id): [] for process_type in process_types for sat_factor in sat_factors for model_id in model_ids}
    
    total_original_time = 0
    total_clipped_time = 0
    total_wrapped_time = 0
    total_recon_time = 0

    for image_name in os.listdir(image_dir):
        if image_name.endswith('.png'):
            image_id = image_name.split('.')[0]
            if any(start <= image_id <= end for start, end in ranges):
                image_path = os.path.join(image_dir, image_name)
                label_path = os.path.join(label_dir, image_name.replace('.png', '.txt'))

                for model_id in model_ids:
                    for sat_factor in sat_factors:
                        annotations, LDR, clipped, wrapped, recon, times = process_image(
                            image_path, label_path, model_id, image_size, conf_threshold, correction, sat_factor, kernel_size, DO, t, vertical
                        )
                        
                        total_original_time += times['original_time']
                        total_clipped_time += times['clipped_time']
                        total_wrapped_time += times['wrapped_time']
                        total_recon_time += times['recon_time']

                        for process_type in process_types:
                            if process_type == "original":
                                metrics = calculate_detection_metrics_1(annotations, LDR)
                            elif process_type == "clip":
                                metrics = calculate_detection_metrics_1(annotations, clipped)
                            elif process_type == "wrap":
                                metrics = calculate_detection_metrics_1(annotations, wrapped)
                            elif process_type == "recon":
                                metrics = calculate_detection_metrics_1(annotations, recon)
                            else:
                                raise ValueError("Invalid process_type. Choose from 'original', 'clip', 'wrap', 'recon'.")

                            all_metrics[(process_type, sat_factor, model_id)].append(metrics)

    global_metrics = {(process_type, sat_factor, model_id): calculate_global_metrics(metrics_list) if metrics_list else {} for (process_type, sat_factor, model_id), metrics_list in all_metrics.items()}

    times = {
        'total_original_time': total_original_time,
        'total_clipped_time': total_clipped_time,
        'total_wrapped_time': total_wrapped_time,
        'total_recon_time': total_recon_time
    }
    for (process_type, sat_factor, model_id), metrics in global_metrics.items():
        save_metrics_to_txt(dataset_dir, image_size, conf_threshold, correction, sat_factor, kernel_size, DO, t, vertical, metrics, process_type, ranges, model_id, times)

    return global_metrics

def save_metrics_to_txt(dataset_dir, image_size, conf_threshold, correction, sat_factor, kernel_size, DO, t, vertical, metrics, process_type, ranges, model_id, times):
    range_str = "_".join([f"{start}_{end}" for start, end in ranges])
    metrics_file_name = f"sat_{sat_factor}_t_{t}_threshold_{conf_threshold}_kernel_{kernel_size}_{process_type}_metrics_{range_str}_{model_id}_workshop.txt"
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
        f.write(f"model_id: {model_id}\n")
        f.write(f"\nImage Ranges Processed: {', '.join([f'{start}-{end}' for start, end in ranges])}\n")
        f.write("\nProcessing Times (seconds):\n")
        f.write(f"Original Time: {times['total_original_time']}\n")
        f.write(f"Clipped Time: {times['total_clipped_time']}\n")
        f.write(f"Wrapped Time: {times['total_wrapped_time']}\n")
        f.write(f"Reconstruction Time: {times['total_recon_time']}\n")
        f.write("\nGlobal Metrics:\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    print(f"Global metrics and configuration parameters saved to {metrics_file_path}")

if __name__ == "__main__":
    dataset_dir = "C:\\Users\\USUARIO\\Documents\\GitHub\\Yolov10\\kitti"
    model_ids = ["yolov10x"]
    image_size = 640
    conf_threshold = 0.60
    correction = 1
    kernel_size = 7
    DO = "1"
    t = 0.6
    vertical = "True"
    ranges = [('007385', '007485')]
    sat_factors = [1.5,2,3]
    process_types = ['clip', 'wrap', 'recon']
    global_metrics = process_dataset(
        dataset_dir, model_ids, image_size, conf_threshold, correction, sat_factors, kernel_size, DO, t, vertical, ranges, process_types
    )
