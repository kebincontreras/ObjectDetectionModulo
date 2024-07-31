from ultralytics import YOLOv10
import os
import torch

def yolov10_inference(image, model_id, image_size, conf_threshold):
    model = YOLOv10.from_pretrained(f'jameslahm/{model_id}')
    results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)
    detections = []
    if results and len(results) > 0:
        for result in results:
            for box in result.boxes:
                detections.append({
                    "coords": box.xyxy.cpu().numpy(),
                    "class": result.names[int(box.cls.cpu())],
                    "conf": box.conf.cpu().numpy()
                    #"bbox": box.xyxy.cpu().numpy().tolist()
                })
    return results[0].plot() if results and len(results) > 0 else image, detections

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou



def calculate_detection_metrics(detections_true, detections_pred):
    true_positives = 0
    pred_positives = len(detections_pred)
    real_positives = len(detections_true)
    ious = []
    for pred in detections_pred:
        for real in detections_true:
            if pred['class'] == real['class']:
                iou = calculate_iou(pred['coords'].flatten(), real['coords'].flatten())
                if iou >= 0.5:
                    true_positives += 1
                    ious.append(iou)
                    break
    precision = true_positives / pred_positives if pred_positives > 0 else 0
    recall = true_positives / real_positives if real_positives > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    average_iou = sum(ious) / len(ious) if ious else 0
    return {"Precision": precision, "Recall": recall, "F1-Score": f1_score, "IOU": average_iou}




def read_kitti_annotations(file_path):
    ground_truths = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if parts[0] != 'DontCare':  # Ignorar 'DontCare'
                class_label = parts[0].lower()  # Clase en minúscula
                bbox = [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]
                ground_truths.append({'class': class_label, 'bbox': bbox})
    return ground_truths


def save_detections(detections, output_dir, filename='detections.txt'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, filename), 'w') as file:
        for detection in detections:
            class_label = detection['class']
            bbox = ','.join(map(str, detection['bbox']))
            file.write(f"{class_label},[{bbox}]\n")


'''
def yolov10_inference_1(image, model_id, image_size, conf_threshold):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLOv10.from_pretrained(f'jameslahm/{model_id}').to(device)
    results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)
    detections = []
    if results and len(results) > 0:
        for result in results:
            for box in result.boxes:
                detections.append({
                    #"coords": box.xyxy.cpu().numpy(),
                    "class": result.names[int(box.cls.cpu())],
                    #"conf": box.conf.cpu().numpy()

                    "bbox": box.xyxy.cpu().numpy().tolist()
                })
    return results[0].plot() if results and len(results) > 0 else image, detections
'''

import time

'''
def yolov10_inference_1(image, model_id, image_size, conf_threshold):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLOv10.from_pretrained(f'jameslahm/{model_id}').to(device)
    start_time = time.time()  # Comienza a medir el tiempo
    results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)
    inference_time = time.time() - start_time  # Calcula el tiempo de inferencia
    detections = []
    if results and len(results) > 0:
        for result in results:
            for box in result.boxes:
                detections.append({
                    "bbox": box.xyxy.cpu().numpy().tolist()
                })
    return results[0].plot() if results and len(results) > 0 else image, detections, inference_time
    #return detections, inference_time
'''
def yolov10_inference_1(image, model_id, image_size, conf_threshold):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLOv10.from_pretrained(f'jameslahm/{model_id}').to(device)
    
    start_time = time.time()  # Comienza a medir el tiempo
    with torch.cuda.amp.autocast(dtype=torch.float16):
	     with torch.no_grad():
              
		      results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)   
    inference_time = time.time() - start_time  # Calcula el tiempo de inferencia
    detections = []
    if results and len(results) > 0:
        for result in results:
            for box in result.boxes:
                detections.append({
                    "bbox": box.xyxy.cpu().numpy().tolist()
                })
    return results[0].plot() if results and len(results) > 0 else image, detections, inference_time
    #return detections, inference_time


def calculate_iou_1(boxA, boxB):
    # Asegúrate de que boxA y boxB son listas planas de flotantes
    # Esto es necesario porque la función max() y min() requieren comparar elementos directamente
    boxA = [float(num) for sublist in boxA for num in sublist] if isinstance(boxA[0], list) else [float(num) for num in boxA]
    boxB = [float(num) for sublist in boxB for num in sublist] if isinstance(boxB[0], list) else [float(num) for num in boxB]

    # Determina las coordenadas de la intersección
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Calcula el área de la intersección
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Calcula el área de cada cuadro delimitador y la unión
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea

    # Calcula el IoU
    iou = interArea / float(unionArea)
    return iou


def calculate_detection_metrics_1(detections_true, detections_pred):
    # Inicializa los contadores para verdaderos positivos, falsos positivos y falsos negativos
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Cuenta el número total de predicciones y de detecciones reales
    pred_positives = len(detections_pred)
    real_positives = len(detections_true)
    
    # Lista para almacenar los valores de IOU (Intersection over Union)
    ious = []
    
    # Itera sobre cada predicción
    for pred in detections_pred:
        pred_bbox = pred['bbox']  # Obtiene la caja delimitadora de la predicción
        pred_class = pred['class']  # Obtiene la clase de la predicción
        matched = False  # Inicializa el indicador de coincidencia
        
        # Itera sobre cada detección real
        for real in detections_true:
            real_bbox = real['bbox']  # Obtiene la caja delimitadora de la detección real
            real_class = real['class']  # Obtiene la clase de la detección real
            
            # Compara si la clase predicha coincide con la clase real
            if pred_class == real_class:
                iou = calculate_iou_1(pred_bbox, real_bbox)  # Calcula el IOU entre la caja predicha y la real
                
                # Si el IOU es mayor o igual a 0.5, cuenta como verdadero positivo
                if iou >= 0.5:
                    true_positives += 1
                    ious.append(iou)  # Añade el IOU a la lista de IOUs
                    matched = True  # Marca la predicción como coincidente
                    break
        
        # Si no hubo coincidencia, cuenta como falso positivo
        if not matched:
            false_positives += 1
    
    # Itera sobre cada detección real para contar los falsos negativos
    for real in detections_true:
        real_bbox = real['bbox']  # Obtiene la caja delimitadora de la detección real
        real_class = real['class']  # Obtiene la clase de la detección real
        matched = False  # Inicializa el indicador de coincidencia
        
        # Itera sobre cada predicción
        for pred in detections_pred:
            pred_bbox = pred['bbox']  # Obtiene la caja delimitadora de la predicción
            pred_class = pred['class']  # Obtiene la clase de la predicción
            
            # Compara si la clase predicha coincide con la clase real
            if pred_class == real_class:
                iou = calculate_iou_1(pred_bbox, real_bbox)  # Calcula el IOU entre la caja predicha y la real
                
                # Si el IOU es mayor o igual a 0.5, cuenta como coincidencia
                if iou >= 0.5:
                    matched = True  # Marca la detección real como coincidente
                    break
        
        # Si no hubo coincidencia, cuenta como falso negativo
        if not matched:
            false_negatives += 1

    # Los verdaderos negativos son difíciles de definir en detección de objetos, así que se deja en cero
    true_negatives = 0
    
    # Calcula la precisión, el recall, el F1-Score y el IOU promedio
    precision = true_positives / pred_positives if pred_positives > 0 else 0
    recall = true_positives / real_positives if real_positives > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    average_iou = sum(ious) / len(ious) if ious else 0

    # Calcula la exactitud
    total_predictions = true_positives + false_positives + false_negatives
    accuracy = (true_positives + true_negatives) / total_predictions if total_predictions > 0 else 0

    # Devuelve un diccionario con las métricas calculadas
    return {
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score,
        "IOU": average_iou,
        "Accuracy": accuracy
    }



