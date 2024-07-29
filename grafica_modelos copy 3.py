import matplotlib.pyplot as plt
import numpy as np

# Datos de la tabla
saturation_levels = [1.5, 2, 3, 4]

iou_s = {
    'Saturate': [56.6, 55.1, 44.4, 34.5],
    'Recovery': [62.0, 61.8, 60.1, 54.2]
}

iou_l = {
    'Saturate': [65.1, 63.0, 57.1, 47.8],
    'Recovery': [69.9, 69.9, 68.4, 60.6]
}

iou_x = {
    'Saturate': [65.5, 63.1, 57.8, 48.8],
    'Recovery': [72.0, 71.2, 70.7, 67.9]
}

f1_s = {
    'Saturate': [36.5, 34.3, 23.5, 16.6],
    'Recovery': [41.6, 41.3, 39.3, 35.1]
}

f1_l = {
    'Saturate': [44.6, 42.3, 33.0, 24.8],
    'Recovery': [52.7, 52.5, 50.3, 42.2]
}

f1_x = {
    'Saturate': [44.9, 42.2, 35.4, 27.1],
    'Recovery': [56.1, 55.7, 52.1, 48.1]
}

acc_s = {
    'Saturate': [27.3, 25.6, 16.3, 11.5],
    'Recovery': [31.6, 32.0, 29.6, 26.0]
}

acc_l = {
    'Saturate': [34.3, 32.1, 23.3, 17.6],
    'Recovery': [42.0, 41.8, 36.1, 31.9]
}

acc_x = {
    'Saturate': [34.3, 31.4, 25.9, 19.3],
    'Recovery': [45.2, 45.1, 42.2, 36.2]
}

# Crear subplots
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 18))

# Subplot para IOU
line10 = axes[0].axhline(y=62.6, color='#3399ff', linestyle='-', label='Ideal HDR')
line11 = axes[0].axhline(y=71.3, color='#0066ff', linestyle='-', label='Ideal HDR')
line12 = axes[0].axhline(y=72.9, color='#0000ff', linestyle='-', label='Ideal HDR')
line1, = axes[0].plot(saturation_levels, iou_s['Saturate'], 'o-', color='#ff6666', label='Saturate')
line3, = axes[0].plot(saturation_levels, iou_s['Recovery'], 'd-.', color='#66ff66', label='Recovery')
line4, = axes[0].plot(saturation_levels, iou_l['Saturate'], '^-', color='#ff3333', label='Saturate')
line6, = axes[0].plot(saturation_levels, iou_l['Recovery'], 'h-.', color='#33cc33', label='Recovery')
line7, = axes[0].plot(saturation_levels, iou_x['Saturate'], 'd-', color='#cc0000', label='Saturate')
line9, = axes[0].plot(saturation_levels, iou_x['Recovery'], 'H-.', color='#009900', label='Recovery')
axes[0].set_xlim([1.5, 4])
axes[0].set_xlabel('Saturation Level (alpha)')
axes[0].set_ylabel('IOU')
first_legend = axes[0].legend(handles=[line10, line3, line1], loc='upper right', title='YOLOv10s')
second_legend = axes[0].legend(handles=[line11, line6, line4], loc='center right', title='YOLOv10l')
third_legend = axes[0].legend(handles=[line12, line9, line7], loc='lower right', title='YOLOv10x')
axes[0].add_artist(first_legend)
axes[0].add_artist(second_legend)

# Subplot para F1-Score
line10 = axes[1].axhline(y=42.1, color='#3399ff', linestyle='-', label='Ideal HDR')
line11 = axes[1].axhline(y=56.3, color='#0066ff', linestyle='-', label='Ideal HDR')
line12 = axes[1].axhline(y=62.3, color='#0000ff', linestyle='-', label='Ideal HDR')
line1, = axes[1].plot(saturation_levels, f1_s['Saturate'], 'o-', color='#ff6666', label='Saturate')
line3, = axes[1].plot(saturation_levels, f1_s['Recovery'], 'd-.', color='#66ff66', label='Recovery')
line4, = axes[1].plot(saturation_levels, f1_l['Saturate'], '^-', color='#ff3333', label='Saturate')
line6, = axes[1].plot(saturation_levels, f1_l['Recovery'], 'h-.', color='#33cc33', label='Recovery')
line7, = axes[1].plot(saturation_levels, f1_x['Saturate'], 'd-', color='#cc0000', label='Saturate')
line9, = axes[1].plot(saturation_levels, f1_x['Recovery'], 'H-.', color='#009900', label='Recovery')
axes[1].set_xlim([1.5, 4])
axes[1].set_xlabel('Saturation Level (alpha)')
axes[1].set_ylabel('F1-Score')
first_legend = axes[1].legend(handles=[line10, line3, line1], loc='upper right', title='YOLOv10s')
second_legend = axes[1].legend(handles=[line11, line6, line4], loc='center right', title='YOLOv10l')
third_legend = axes[1].legend(handles=[line12, line9, line7], loc='lower right', title='YOLOv10x')
axes[1].add_artist(first_legend)
axes[1].add_artist(second_legend)

# Subplot para Accuracy
line10 = axes[2].axhline(y=32.1, color='#3399ff', linestyle='-', label='Ideal HDR')
line11 = axes[2].axhline(y=45.5, color='#0066ff', linestyle='-', label='Ideal HDR')
line12 = axes[2].axhline(y=50.2, color='#0000ff', linestyle='-', label='Ideal HDR')
line1, = axes[2].plot(saturation_levels, acc_s['Saturate'], 'o-', color='#ff6666', label='Saturate')
line3, = axes[2].plot(saturation_levels, acc_s['Recovery'], 'd-.', color='#66ff66', label='Recovery')
line4, = axes[2].plot(saturation_levels, acc_l['Saturate'], '^-', color='#ff3333', label='Saturate')
line6, = axes[2].plot(saturation_levels, acc_l['Recovery'], 'h-.', color='#33cc33', label='Recovery')
line7, = axes[2].plot(saturation_levels, acc_x['Saturate'], 'd-', color='#cc0000', label='Saturate')
line9, = axes[2].plot(saturation_levels, acc_x['Recovery'], 'H-.', color='#009900', label='Recovery')
axes[2].set_xlim([1.5, 4])
axes[2].set_xlabel('Saturation Level (alpha)')
axes[2].set_ylabel('Accuracy')
first_legend = axes[2].legend(handles=[line10, line3, line1], loc='upper right', title='YOLOv10s')
second_legend = axes[2].legend(handles=[line11, line6, line4], loc='center right', title='YOLOv10l')
third_legend = axes[2].legend(handles=[line12, line9, line7], loc='lower right', title='YOLOv10x')
axes[2].add_artist(first_legend)
axes[2].add_artist(second_legend)

# Ajustar diseño
plt.tight_layout()
plt.show()
