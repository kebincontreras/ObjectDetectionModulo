import os
import time
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from utils import modulo
from utils_spud import recons_spud

# Definición del dispositivo para la aceleración por GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def process_image(image_path, sat_factor):
    # Cargar imagen
    image = Image.open(image_path)
    original_image = np.array(image)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)  # Convertir de RGB a BGR
    original_image = original_image - original_image.min()
    original_image = original_image / original_image.max() * 255.0
    original_image = original_image.astype(np.uint8)

    # Convertir imagen a tensor
    img_tensor = torch.tensor(original_image / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    img_tensor = modulo(img_tensor * sat_factor, L=1.0)

    # Medir el tiempo de procesamiento del SPUD
    start_time = time.time()
    recon_image = recons_spud(img_tensor, threshold=0.1, mx=1.0)
    processing_time = time.time() - start_time

    # Convertir tensor de imagen reconstruida a formato NumPy
    recon_image_np = recon_image.squeeze().cpu().numpy().transpose(1, 2, 0)
    recon_image_np = (recon_image_np * 255).astype(np.uint8)

    return original_image, recon_image_np, processing_time * 1000  # Tiempo en milisegundos

def plot_images(original_images, recon_images, processing_times, image_ids):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('SPUD Reconstruction and Processing Time (ms)', fontsize=16)

    for i in range(3):
        # Mostrar imagen original
        axes[0, i].imshow(cv2.cvtColor(original_images[i], cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(f'Original {image_ids[i]}')
        axes[0, i].axis('off')

        # Mostrar imagen reconstruida
        axes[1, i].imshow(cv2.cvtColor(recon_images[i], cv2.COLOR_BGR2RGB))
        axes[1, i].set_title(f'Reconstructed {image_ids[i]} \nTime: {processing_times[i]:.2f} ms')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def process_image_range(dataset_dir, start_image, end_image, sat_factor):
    image_dir = os.path.join(dataset_dir, 'image')
    image_range = range(int(start_image), int(end_image) + 1)

    original_images = []
    recon_images = []
    processing_times = []
    image_ids = []

    for image_id in image_range:
        image_name = f"{image_id:06d}.png"  # Formato de nombre de imagen
        image_path = os.path.join(image_dir, image_name)

        if os.path.exists(image_path):
            original, recon, processing_time = process_image(image_path, sat_factor)
            original_images.append(original)
            recon_images.append(recon)
            processing_times.append(processing_time)
            image_ids.append(image_name)
            print(f"Processed image: {image_name}, SPUD processing time: {processing_time:.2f} milliseconds")
        else:
            print(f"Image {image_name} not found in the directory.")

    plot_images(original_images, recon_images, processing_times, image_ids)

if __name__ == "__main__":
    dataset_dir = "C:\\Users\\USUARIO\\Documents\\GitHub\\Yolov10\\kitti"
    start_image = "000000"  # Imagen de inicio del rango
    end_image = "000009"    # Imagen final del rango
    sat_factor = 2.0        # Factor de saturación utilizado para SPUD

    process_image_range(dataset_dir, start_image, end_image, sat_factor)
