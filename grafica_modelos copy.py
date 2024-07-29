import matplotlib.pyplot as plt
import numpy as np

# Datos de la tabla
saturation_levels = [1.5, 2, 3, 4]

iou_s = {
    'Saturate': [56.6, 55.1, 44.4, 34.5],
    'Modulo': [61.8, 60.1, 57.6, 54.3],
    'Recovery': [62.0, 61.8, 60.1, 54.2]
}

iou_l = {
    'Saturate': [65.1, 63.0, 57.1, 47.8],
    'Modulo': [69.4, 68.7, 65.4, 58.8],
    'Recovery': [69.9, 69.9, 68.4, 60.6]
}

iou_x = {
    'Saturate': [65.5, 63.1, 57.8, 48.8],
    'Modulo': [71.0, 70.0, 67.9, 62.3],
    'Recovery': [72.0, 71.2, 70.7, 67.9]
}

# Crear la gráfica
plt.figure(figsize=(12, 6))

# Colores vivos
color_saturate = 'red'
color_modulo = 'orange'
color_recovery = 'green'
color_hdr = 'blue'

# YOLOv10s
line1, = plt.plot(saturation_levels, iou_s['Saturate'], 'o-', color='#ff6666', label='YOLOv10s Saturate')
line2, = plt.plot(saturation_levels, iou_s['Modulo'], 's--', color='#ffcc66', label='YOLOv10s Modulo')
line3, = plt.plot(saturation_levels, iou_s['Recovery'], 'd-.', color='#66ff66', label='YOLOv10s Recovery')

# YOLOv10l
line4, = plt.plot(saturation_levels, iou_l['Saturate'], '^-', color='#ff3333', label='YOLOv10l Saturate')
line5, = plt.plot(saturation_levels, iou_l['Modulo'], 'p--', color='#ff9933', label='YOLOv10l Modulo')
line6, = plt.plot(saturation_levels, iou_l['Recovery'], 'h-.', color='#33cc33', label='YOLOv10l Recovery')

# YOLOv10x
line7, = plt.plot(saturation_levels, iou_x['Saturate'], 'd-', color='#cc0000', label='YOLOv10x Saturate')
line8, = plt.plot(saturation_levels, iou_x['Modulo'], '*--', color='#ff6600', label='YOLOv10x Modulo')
line9, = plt.plot(saturation_levels, iou_x['Recovery'], 'H-.', color='#009900', label='YOLOv10x Recovery')

# Ideal HDR
line10 = plt.axhline(y=62.6, color='#3399ff', linestyle='-', label='Ideal HDR (YOLOv10s)')
line11 = plt.axhline(y=71.3, color='#0066ff', linestyle='-', label='Ideal HDR (YOLOv10l)')
line12 = plt.axhline(y=72.9, color='#0000ff', linestyle='-', label='Ideal HDR (YOLOv10x)')

# Configuraciones de la gráfica
plt.xlabel('Saturation Level (alpha)')
plt.ylabel('Performance (IOU)')
plt.title('Performance Metrics vs Saturation Level for YOLOv10 Models')

# Agrupar las leyendas
first_legend = plt.legend(handles=[line1, line2, line3, line10], loc='center left', bbox_to_anchor=(1, 0.75), title='YOLOv10s')
second_legend = plt.legend(handles=[line4, line5, line6, line11], loc='center left', bbox_to_anchor=(1, 0.5), title='YOLOv10l')
third_legend = plt.legend(handles=[line7, line8, line9, line12], loc='center left', bbox_to_anchor=(1, 0.25), title='YOLOv10x')

# Añadir las leyendas al gráfico
plt.gca().add_artist(first_legend)
plt.gca().add_artist(second_legend)

plt.grid(True)
plt.tight_layout()  # Ajusta el diseño para que la leyenda se muestre fuera de la gráfica
plt.show()
