
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
    true_positives = 0
    pred_positives = len(detections_pred)
    real_positives = len(detections_true)
    ious = []
    for pred in detections_pred:
        pred_bbox = pred['bbox']
        pred_class = pred['class']
        for real in detections_true:
            real_bbox = real['bbox']
            real_class = real['class']
            if pred_class == real_class:
                iou = calculate_iou_1(pred_bbox, real_bbox)
                if iou >= 0.5:
                    true_positives += 1
                    ious.append(iou)
                    break
    precision = true_positives / pred_positives if pred_positives > 0 else 0
    recall = true_positives / real_positives if real_positives > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    average_iou = sum(ious) / len(ious) if ious else 0
    return {"Precision": precision, "Recall": recall, "F1-Score": f1_score, "IOU": average_iou}


