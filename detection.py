
from ultralytics import YOLOv10

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
