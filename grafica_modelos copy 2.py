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

# Crear la gráfica
plt.figure(figsize=(12, 6))

# Colores vivos
color_saturate = 'red'
color_recovery = 'green'
color_hdr = 'blue'

# YOLOv10s
line1, = plt.plot(saturation_levels, iou_s['Saturate'], 'o-', color='#ff6666', label='YOLOv10s Saturate')
line3, = plt.plot(saturation_levels, iou_s['Recovery'], 'd-.', color='#66ff66', label='YOLOv10s Recovery')

# YOLOv10l
line4, = plt.plot(saturation_levels, iou_l['Saturate'], '^-', color='#ff3333', label='YOLOv10l Saturate')
line6, = plt.plot(saturation_levels, iou_l['Recovery'], 'h-.', color='#33cc33', label='YOLOv10l Recovery')

# YOLOv10x
line7, = plt.plot(saturation_levels, iou_x['Saturate'], 'd-', color='#cc0000', label='YOLOv10x Saturate')
line9, = plt.plot(saturation_levels, iou_x['Recovery'], 'H-.', color='#009900', label='YOLOv10x Recovery')

# Ideal HDR
line10 = plt.axhline(y=62.6, color='#3399ff', linestyle='-', label='Ideal HDR')
line11 = plt.axhline(y=71.3, color='#0066ff', linestyle='-', label='Ideal HDR')
line12 = plt.axhline(y=72.9, color='#0000ff', linestyle='-', label='Ideal HDR')

# Configuraciones de la gráfica
plt.xlabel('Saturation Level (alpha)')
plt.ylabel('Performance (IOU)')
plt.title('Performance Metrics vs Saturation Level for YOLOv10 Models')

# Agrupar las leyendas
first_legend = plt.legend(handles=[line10, line3, line1], loc='center left', bbox_to_anchor=(1, 0.75), title='YOLOv10s')
second_legend = plt.legend(handles=[line11, line6, line4], loc='center left', bbox_to_anchor=(1, 0.5), title='YOLOv10l')
third_legend = plt.legend(handles=[line12, line9, line7], loc='center left', bbox_to_anchor=(1, 0.25), title='YOLOv10x')

# Añadir las leyendas al gráfico
plt.gca().add_artist(first_legend)
plt.gca().add_artist(second_legend)

plt.grid(True)
plt.tight_layout()  # Ajusta el diseño para que la leyenda se muestre fuera de la gráfica
plt.show()
