import os
import pandas as pd

# Define la ruta base y las carpetas
base_path = r"C:\Users\USUARIO\Documents\GitHub\Yolov10\kitti\Results model WorkShop"
folders = ["yolov10B", "yolov10L", "yolov10M", "yolov10N", "yolov10S", "yolov10X"]

# Lista para almacenar los datos
data = []

# Función para extraer el tipo de archivo del nombre
def extract_file_type(file_name):
    if "clip" in file_name:
        return "clip"
    elif "original" in file_name:
        return "original"
    elif "recon" in file_name:
        return "recon"
    elif "wrap" in file_name:
        return "wrap"
    else:
        return "unknown"

# Recorre cada carpeta y archivo .txt
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r") as file:
                content = file.read()
                params = {
                    "model_id": None,
                    "sat_factor": None,
                    "file_type": extract_file_type(file_name),
                    "Precision": None,
                    "Recall": None,
                    "F1-Score": None,
                    "IOU": None,
                    "Accuracy": None
                }
                
                for line in content.split('\n'):
                    if line.startswith("model_id:"):
                        params["model_id"] = line.split(': ')[1]
                    elif line.startswith("sat_factor:"):
                        params["sat_factor"] = line.split(': ')[1]
                    elif line.startswith("Precision:"):
                        params["Precision"] = float(line.split(': ')[1]) * 100
                    elif line.startswith("Recall:"):
                        params["Recall"] = float(line.split(': ')[1]) * 100
                    elif line.startswith("F1-Score:"):
                        params["F1-Score"] = float(line.split(': ')[1]) * 100
                    elif line.startswith("IOU:"):
                        params["IOU"] = float(line.split(': ')[1]) * 100
                    elif line.startswith("Accuracy:"):
                        params["Accuracy"] = float(line.split(': ')[1]) * 100

                data.append(params)

# Crear DataFrame
df = pd.DataFrame(data)

# Guardar el DataFrame en un archivo Excel
output_file = os.path.join(base_path, "summary_metrics.xlsx")
df.to_excel(output_file, index=False)

print(f"Archivo Excel guardado en: {output_file}")
