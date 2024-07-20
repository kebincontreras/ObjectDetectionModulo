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
    # Inicializa un diccionario vacío para acumular las sumas de cada métrica
    metrics_sum = {}
    
    # Itera sobre cada diccionario de métricas en la lista de métricas
    for metrics in metrics_list:
        # Itera sobre cada par clave-valor en el diccionario de métricas
        for key, value in metrics.items():
            # Si la clave no está en el diccionario de sumas, inicialízala a 0
            if key not in metrics_sum:
                metrics_sum[key] = 0
            # Suma el valor de la métrica actual a la suma acumulada para esa clave
            metrics_sum[key] += value
    
    # Calcula la cantidad de diccionarios de métricas
    metrics_count = len(metrics_list)
    
    # Calcula la media de cada métrica dividiendo la suma acumulada entre la cantidad de diccionarios
    mean_metrics = {key: value / metrics_count for key, value in metrics_sum.items()}
    
    # Devuelve un diccionario con las métricas promediadas
    return mean_metrics



def process_image(image_path, annotations_path, model_id, image_size, conf_threshold, correction, sat_factor, kernel_size, DO, t, vertical):
    image = Image.open(image_path)
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

    img_tensor = torch.tensor(blurred_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    img_tensor = modulo(img_tensor * sat_factor, L=1.0)

    wrapped_image = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    wrapped_image = (wrapped_image * 255).astype(np.uint8)

    original_annotated, LDR_detections = yolov10_inference_1(original_image, model_id, image_size, conf_threshold)
    #original_annotated, LDR_detections = yolov10_inference_1(blurred_image, model_id, image_size, conf_threshold)
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

    #image_id = os.path.splitext(os.path.basename(image_path))[0]  # Extrae el identificador de la imagen sin la extensión
    #image_dir = os.path.dirname(image_path)  # Obtiene el directorio de la imagen
    #save_images(image_dir, image_id, original_image, clipped_image, wrapped_image, recon_image_np)
    

    return original_annotations, LDR_detections, clipped_detections, wrapped_detections, recon_detections


def process_dataset(dataset_dir, model_id, image_size, conf_threshold, correction, sat_factor, kernel_size, DO, t, vertical):
    # Define los directorios de imágenes y etiquetas dentro del directorio del dataset
    image_dir = os.path.join(dataset_dir, 'image')
    label_dir = os.path.join(dataset_dir, 'label')

    # Inicializa listas para almacenar métricas de detección para diferentes métodos de procesamiento de imágenes
    all_metrics_orig = []
    all_metrics_clip = []
    all_metrics_wrap = []
    all_metrics_recons = []

    # Itera sobre cada nombre de archivo en el directorio de imágenes
    for image_name in os.listdir(image_dir):
        if image_name.endswith('.png'):  # Procesa solo archivos PNG
            image_path = os.path.join(image_dir, image_name)  # Obtiene la ruta completa de la imagen
            label_path = os.path.join(label_dir, image_name.replace('.png', '.txt'))  # Obtiene la ruta completa de la etiqueta correspondiente

            # Procesa la imagen y obtiene las anotaciones y las detecciones para diferentes métodos
            original_annotations, LDR_detections, clipped_detections, wrapped_detections, recon_detections = process_image(
                image_path, label_path, model_id, image_size, conf_threshold, correction, sat_factor, kernel_size, DO, t, vertical
            )

            # Calcula las métricas de detección para las anotaciones originales y las detecciones procesadas
            metrics_orig = calculate_detection_metrics_1(original_annotations, LDR_detections)
            metrics_clip = calculate_detection_metrics_1(original_annotations, clipped_detections)
            metrics_wrap = calculate_detection_metrics_1(original_annotations, wrapped_detections)
            metrics_recons = calculate_detection_metrics_1(original_annotations, recon_detections)

            # Añade las métricas calculadas a las listas correspondientes
            all_metrics_orig.append(metrics_orig)
            all_metrics_clip.append(metrics_clip)
            all_metrics_wrap.append(metrics_wrap)
            all_metrics_recons.append(metrics_recons)

    # Calcula las métricas globales promediadas de las listas anteriores para cada método 
    global_metrics_orig = calculate_global_metrics(all_metrics_orig)
    global_metrics_clip = calculate_global_metrics(all_metrics_clip)
    global_metrics_wrap = calculate_global_metrics(all_metrics_wrap)
    global_metrics_recons = calculate_global_metrics(all_metrics_recons)


    save_metrics_to_txt(dataset_dir, image_size, conf_threshold, correction, sat_factor, kernel_size, DO, t, vertical, global_metrics_orig, global_metrics_clip, global_metrics_wrap, global_metrics_recons)

    return global_metrics_orig, global_metrics_clip, global_metrics_wrap, global_metrics_recons

 
def save_metrics_to_txt(dataset_dir, image_size, conf_threshold, correction, sat_factor, kernel_size, DO, t, vertical, metrics_orig, metrics_clip, metrics_wrap, metrics_recons):
    # Construct the filename using the configuration parameters
    #metrics_file_name = f"sat_{sat_factor}_t_{t}_threshold_{conf_threshold}_kernel_{kernel_size}_7481image_kebin1_final_19_07_2024_final.txt"
    metrics_file_name = f"prueba_eliminar"
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
        f.write("\nGlobal Metrics for Original Images:\n")
        for key, value in metrics_orig.items():
            f.write(f"{key}: {value}\n")
        f.write("\nGlobal Metrics for Clipped Images:\n")
        for key, value in metrics_clip.items():
            f.write(f"{key}: {value}\n")
        f.write("\nGlobal Metrics for Wrapped Images:\n")
        for key, value in metrics_wrap.items():
            f.write(f"{key}: {value}\n")
        f.write("\nGlobal Metrics for Reconstructed Images:\n")
        for key, value in metrics_recons.items():
            f.write(f"{key}: {value}\n")
    print(f"Global metrics and configuration parameters saved to {metrics_file_path}")


if __name__ == "__main__":
    #dataset_dir = "C:\\Users\\USUARIO\\Documents\\GitHub\\Yolov10\\kitti"
    dataset_dir = "C:\\Users\\USUARIO\\Desktop\\dataset_eliminar"
    model_id = "yolov10x"
    image_size = 640
    conf_threshold = 0.60
    correction = 1
    sat_factor = 5
    kernel_size = 7
    DO = "1"
    t = 0.6
    vertical = "True"

    global_metrics_orig, global_metrics_clip, global_metrics_wrap, global_metrics_recons = process_dataset(
        dataset_dir, model_id, image_size, conf_threshold, correction, sat_factor, kernel_size, DO, t, vertical
    )

    # Print global metrics
    print("Global Metrics for Original Images:", global_metrics_orig)
    print("Global Metrics for Clipped Images:", global_metrics_clip)
    print("Global Metrics for Wrapped Images:", global_metrics_wrap)
    print("Global Metrics for Reconstructed Images:", global_metrics_recons)
