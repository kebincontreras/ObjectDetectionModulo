
'''
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


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Suponiendo que estas funciones ya están definidas correctamente:
from detection import yolov10_inference, calculate_detection_metrics
from image_processing import apply_blur, clip_image, wrap_image

# Nueva función para cargar las etiquetas de KITTI
def load_kitti_labels(label_path):
    labels = []
    with open(label_path, 'r') as file:
        for line in file:
            labels.append(line.strip().split())
    return labels

# Actualización de la función process_image para incluir las etiquetas
def process_image(image_path, label_path, model_id, image_size, conf_threshold):
    image = Image.open(image_path)
    original_image = np.array(image)
    
    # Cargar etiquetas KITTI
    kitti_labels = load_kitti_labels(label_path)
    
    # Procesamiento de la imagen
    blurred_image = apply_blur(original_image, kernel_size=3)
    clipped_image = clip_image(blurred_image, correction=0.5, sat_factor=2.0)
    wrapped_image = wrap_image(blurred_image, correction=0.5, sat_factor=2.0)
    
    # Detección de objetos
    detections = yolov10_inference(original_image, model_id, image_size, conf_threshold)
    
    # Cálculo de métricas utilizando las etiquetas de KITTI
    metrics = calculate_detection_metrics(detections, kitti_labels)

    return original_image, clipped_image, wrapped_image, detections, metrics

# Ejemplo de cómo llamar a la función
if __name__ == "__main__":
    image_path = "C:\\Users\\USUARIO\\Documents\\GitHub\\Yolov10\\kitti\\image_2\\000000.png"
    label_path = "C:\\Users\\USUARIO\\Documents\\GitHub\\Yolov10\\kitti\\label_2\\000000.txt"
    model_id = "yolov10m"
    image_size = 640
    conf_threshold = 0.85
    result = process_image(image_path, label_path, model_id, image_size, conf_threshold)
    
    original_image, clipped_image, wrapped_image, detections, metrics = result
    
    # Mostrar las imágenes procesadas
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.subplot(132)
    plt.imshow(clipped_image)
    plt.title('Clipped Image')
    plt.subplot(133)
    plt.imshow(wrapped_image)
    plt.title('Wrapped Image')
    plt.show()
    
    # Imprimir las métricas
    print("Detection Metrics:", metrics)

'''


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


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Suponiendo que estas funciones ya están definidas correctamente:
from detection import yolov10_inference, calculate_detection_metrics
from image_processing import apply_blur, clip_image, wrap_image

# Nueva función para cargar las etiquetas de KITTI
def load_kitti_labels(label_path):
    labels = []
    with open(label_path, 'r') as file:
        for line in file:
            labels.append(line.strip().split())
    return labels

# Actualización de la función process_image para incluir las etiquetas
def process_image(image_path, label_path, model_id, image_size, conf_threshold):
    image = Image.open(image_path)
    original_image = np.array(image)
    
    # Cargar etiquetas KITTI
    kitti_labels = load_kitti_labels(label_path)
    
    # Procesamiento de la imagen
    blurred_image = apply_blur(original_image, kernel_size=3)
    clipped_image = clip_image(blurred_image, correction=0.5, sat_factor=2.0)
    wrapped_image = wrap_image(blurred_image, correction=0.5, sat_factor=2.0)
    
    # Detección de objetos
    detections = yolov10_inference(original_image, model_id, image_size, conf_threshold)
    
    # Cálculo de métricas utilizando las etiquetas de KITTI
    #metrics = calculate_detection_metrics(detections, kitti_labels)

    return original_image, clipped_image, wrapped_image, detections

# Ejemplo de cómo llamar a la función
if __name__ == "__main__":
    image_path = "C:\\Users\\USUARIO\\Documents\\GitHub\\Yolov10\\kitti\\image_2\\000000.png"
    label_path = "C:\\Users\\USUARIO\\Documents\\GitHub\\Yolov10\\kitti\\label_2\\000000.txt"
    model_id = "yolov10m"
    image_size = 640
    conf_threshold = 0.85
    result = process_image(image_path, label_path, model_id, image_size, conf_threshold)
    
    original_image, clipped_image, wrapped_image, detections = result
    
    # Mostrar las imágenes procesadas
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.subplot(132)
    plt.imshow(clipped_image)
    plt.title('Clipped Image')
    plt.subplot(133)
    plt.imshow(wrapped_image)
    plt.title('Wrapped Image')
    plt.show()
