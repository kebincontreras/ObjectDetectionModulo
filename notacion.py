import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# Mapeo de clases de YOLOv10 a las clases de tu base de datos


def map_yolo_classes_to_db(yolo_detections):
    """
    Ajusta las clases detectadas por YOLO a las clases de la base de datos según el mapeo proporcionado.
    """
    mapping = {
    'person': 'pedestrian',
    'bicycle': 'cyclist',
    'car': 'car',
    'motorbike': 'cisc',
    'aeroplane': 'cisc',
    'bus': 'cisc',
    'train': 'tram',
    'truck': 'truck',
    'van': 'van'
     }
    mapped_detections = []
    for det in yolo_detections:
        if det['class'] in mapping:
            det['class'] = mapping[det['class']]
            mapped_detections.append(det)
    return mapped_detections

