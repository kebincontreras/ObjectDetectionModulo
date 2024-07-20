import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# Mapeo de clases de YOLOv10 a las clases de tu base de datos


def map_yolo_classes_to_db(yolo_detections):
    """
    Ajusta las clases detectadas por YOLO a las clases de la base de datos según el mapeo proporcionado.
    """
    
    #mapping = {'person': 'pedestrian','bicycle': 'cyclist','car': 'car','motorbike': 'cisc','aeroplane': 'cisc','bus': 'cisc','train': 'tram',
    #'truck': 'truck','van': 'van'
    # }
        
    mapping = {'person': 'pedestrian', 'bicycle': 'cyclist', 'car': 'car', 'motorbike': 'cyclist', 'aeroplane': 'misc', 'bus': 'tram', 'train': 'tram', 'truck': 'truck', 'van': 'van', 'boat': 'misc', 'traffic light': 'dontcare', 'fire hydrant': 'dontCare', 'stop sign': 'DdontCare', 'parking meter': 'dontCare', 'bench': 'dontCare', 'bird': 'dontCare', 'cat': 'dontCare', 'dog': 'dontCare', 'horse': 'dontCare', 'sheep': 'dontCare', 'cow': 'dontCare', 'elephant': 'dontCare', 'bear': 'dontCare', 'zebra': 'dontCare', 'giraffe': 'dontCare', 'backpack': 'dontCare', 'umbrella': 'dontCare', 'handbag': 'dontCare', 'tie': 'dontCare', 'suitcase': 'dontCare', 'frisbee': 'dontCare', 'skis': 'dontCare', 'snowboard': 'dontCare', 'sports ball': 'dontCare', 'kite': 'dontCare', 'baseball bat': 'DontCare', 'baseball glove': 'DontCare', 'skateboard': 'DontCare', 'surfboard': 'DontCare', 'tennis racket': 'DontCare', 'bottle': 'DontCare', 'wine glass': 'DontCare', 'cup': 'DontCare', 'fork': 'DontCare', 'knife': 'DontCare', 'spoon': 'DontCare', 'bowl': 'DontCare', 'banana': 'DontCare', 'apple': 'DontCare', 'sandwich': 'DontCare', 'orange': 'DontCare', 'broccoli': 'DontCare', 'carrot': 'DontCare', 'hot dog': 'DontCare', 'pizza': 'DontCare', 'donut': 'DontCare', 'cake': 'DontCare', 'chair': 'DontCare', 'sofa': 'DontCare', 'pottedplant': 'DontCare', 'bed': 'DontCare', 'diningtable': 'DontCare', 'toilet': 'DontCare', 'tvmonitor': 'DontCare', 'laptop': 'DontCare', 'mouse': 'DontCare', 'remote': 'DontCare', 'keyboard': 'DontCare', 'cell phone': 'DontCare', 'microwave': 'DontCare', 'oven': 'DontCare', 'toaster': 'DontCare', 'sink': 'DontCare', 'refrigerator': 'DontCare', 'book': 'DontCare', 'clock': 'DontCare', 'vase': 'DontCare', 'scissors': 'DontCare', 'teddy bear': 'DontCare', 'hair drier': 'DontCare', 'toothbrush': 'DontCare'}

    mapped_detections = []
    for det in yolo_detections:
        if det['class'] in mapping:
            det['class'] = mapping[det['class']]
            mapped_detections.append(det)
    return mapped_detections

