import os
import time
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms
from image_processing import apply_blur, clip_image, wrap_image
from ultralytics import YOLOv10  # Asegúrate de que esta es la importación correcta
from utils import modulo
from utils_spud import recons_spud

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def yolov10_inference_1(image, model_id, image_size, conf_threshold):
    model = YOLOv10.from_pretrained(f'jameslahm/{model_id}').to(device)
    
    start_time = time.time()
    with torch.cuda.amp.autocast(dtype=torch.float16):
        with torch.no_grad():
            results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)
    inference_time = time.time() - start_time  # Tiempo en segundos
    
    return results[0].plot() if results and len(results) > 0 else image, [], inference_time * 1000  # Convertir a milisegundos

def process_image(image_path, image_size, conf_threshold, correction, sat_factor, kernel_size):
    image = Image.open(image_path)
    original_image = np.array(image)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min()) * 255.0
    original_image = original_image.astype(np.uint8)

    blurred_image = apply_blur(original_image, kernel_size)
    clipped_image = clip_image(blurred_image, correction, sat_factor)

    img_tensor = torch.tensor(blurred_image / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    img_tensor = modulo(img_tensor * sat_factor, L=1.0)

    wrapped_image = (img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    _, _, original_time = yolov10_inference_1(original_image, 'yolov10n', image_size, conf_threshold)
    _, _, clipped_time = yolov10_inference_1(clipped_image, 'yolov10n', image_size, conf_threshold)
    _, _, wrapped_time = yolov10_inference_1(wrapped_image, 'yolov10n', image_size, conf_threshold)

    # Medir tiempo de ejecución de recons_spud
    start_recon_spud_time = time.time()
    recon_image = recons_spud(img_tensor, threshold=0.1, mx=1.0)
    recon_spud_time = (time.time() - start_recon_spud_time) * 1000  # Convertir a milisegundos

    recon_image_pil = transforms.ToPILImage()(recon_image.squeeze(0).cpu())
    recon_image_np = np.array(recon_image_pil).astype(np.uint8)
    _, _, recon_infer_time = yolov10_inference_1(recon_image_np, 'yolov10n', image_size, conf_threshold)
    total_recon_time = recon_spud_time + recon_infer_time

    return {
        'original': original_time,
        'clip': clipped_time,
        'wrap': wrapped_time,
        'recon': total_recon_time
    }

def process_dataset(dataset_dir, image_size, conf_threshold, correction, sat_factor, kernel_size, ranges):
    image_dir = os.path.join(dataset_dir, 'image')
    times_accumulated = {process_type: [] for process_type in ['original', 'clip', 'wrap', 'recon']}

    for image_name in os.listdir(image_dir):
        if image_name.endswith('.png'):
            image_id = image_name.split('.')[0]
            if any(start <= image_id <= end for start, end in ranges):
                image_path = os.path.join(image_dir, image_name)
                process_results = process_image(image_path, image_size, conf_threshold, correction, sat_factor, kernel_size)
                for process_type, time_taken in process_results.items():
                    times_accumulated[process_type].append(time_taken)
    
    for process_type, times in times_accumulated.items():
        average_time = np.mean(times)
        std_dev = np.std(times)
        print(f"{process_type.capitalize()} - Average Time: {average_time:.2f} ms, Standard Deviation: {std_dev:.2f} ms")

if __name__ == "__main__":
    dataset_dir = "C:\\Users\\USUARIO\\Documents\\GitHub\\Yolov10\\kitti"
    image_size = 640
    conf_threshold = 0.60
    correction = 1
    kernel_size = 7
    sat_factor = 2
    ranges = [('007480', '007485')]  # Rango de imágenes a procesar

    process_dataset(dataset_dir, image_size, conf_threshold, correction, sat_factor, kernel_size, ranges)
