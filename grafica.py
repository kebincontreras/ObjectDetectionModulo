

import pandas as pd
import matplotlib.pyplot as plt

# Updated data for IOU as provided by the user
updated_data_iou = {
    'Model': ['YOLOv10s', 'YOLOv10s', 'YOLOv10s', 'YOLOv10l', 'YOLOv10l', 'YOLOv10l', 'YOLOv10x', 'YOLOv10x', 'YOLOv10x'],
    'Type': ['Saturate', 'Modulo', 'Recovery', 'Saturate', 'Modulo', 'Recovery', 'Saturate', 'Modulo', 'Recovery'],
    'α1.5': [56.6, 61.8, 62.0, 65.1, 69.4, 69.9, 65.5, 71.0, 72.0],
    'α2.0': [55.1, 60.1, 61.8, 63.0, 68.7, 69.9, 63.1, 70.0, 71.2],
    'α3.0': [44.4, 57.6, 60.1, 57.1, 65.4, 68.4, 57.8, 67.9, 70.7],
    'α4.0': [34.5, 54.3, 54.2, 47.8, 58.8, 60.6, 48.8, 62.3, 67.9],
    'Ideal HDR': [62.6, 62.6, 62.6, 71.3, 71.3, 71.3, 72.9, 72.9, 72.9]
}

# Define the order of models
models_order = ['YOLOv10s', 'YOLOv10l', 'YOLOv10x']

# Convert to DataFrame
df_updated_iou = pd.DataFrame(updated_data_iou)

# Melt the DataFrame for easier plotting
df_updated_iou_melted = df_updated_iou.melt(id_vars=['Model', 'Type'], var_name='Saturation Level', value_name='Performance')

# Define the updated vibrant colors, including Ideal HDR
vibrant_colors_custom = {
    'Ideal HDR': '#6495ED',  # Cornflower Blue for Ideal HDR
    'α1.5': '#32CD32',  # Green
    'α2.0': '#FFD700',  # Yellow
    'α3.0': '#FF69B4',  # Pink
    'α4.0': '#FF6347',  # Red
}

# Define the order of types
order = ['Ideal HDR', 'Recovery', 'Modulo', 'Saturate']

# Plotting with vibrant colors for updated IOU data and only one legend
fig, ax = plt.subplots(figsize=(15, 6))

# Bar plot with vibrant colors for updated IOU data
for i, model in enumerate(models_order):
    ax = plt.subplot(1, 3, i + 1)
    group = df_updated_iou_melted[df_updated_iou_melted['Model'] == model]
    group_pivot = group.pivot(index='Type', columns='Saturation Level', values='Performance')
    group_pivot = group_pivot.reindex(order[1:])  # Skip Ideal HDR for x-axis
    
    colors = [vibrant_colors_custom[col] for col in group_pivot.columns]
    group_pivot.plot(kind='bar', ax=ax, color=colors)
    
    ax.set_title(model)
    ax.set_ylabel('IOU')
    ax.set_xlabel('')
    ax.set_ylim(30, 75)
    ax.set_xticks(range(len(group_pivot.index)))
    ax.set_xticklabels(group_pivot.index, rotation=45)
    if i == 0:
        ax.legend(title='Saturation Level')
    else:
        ax.get_legend().remove()
    ax.set_yticks(range(30, 76, 5))

plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt

# Updated data for F1-Score as provided by the user
updated_data_f1 = {
    'Model': ['YOLOv10s', 'YOLOv10s', 'YOLOv10s', 'YOLOv10l', 'YOLOv10l', 'YOLOv10l', 'YOLOv10x', 'YOLOv10x', 'YOLOv10x'],
    'Type': ['Saturate', 'Modulo', 'Recovery', 'Saturate', 'Modulo', 'Recovery', 'Saturate', 'Modulo', 'Recovery'],
    'α1.5': [36.5, 42.2, 41.6, 44.6, 53.3, 52.7, 44.9, 54.8, 56.1],
    'α2.0': [34.3, 39.3, 41.3, 42.3, 53.3, 52.5, 42.2, 54.1, 55.7],
    'α3.0': [23.5, 36.0, 39.3, 33.0, 46.2, 50.3, 35.4, 51.7, 52.1],
    'α4.0': [16.6, 31.1, 35.1, 24.8, 38.4, 42.2, 27.1, 43.9, 48.1],
    'Ideal HDR': [42.1, 42.1, 42.1, 56.3, 56.3, 56.3, 62.3, 62.3, 62.3]
}

# Convert to DataFrame
df_updated_f1 = pd.DataFrame(updated_data_f1)

# Melt the DataFrame for easier plotting
df_updated_f1_melted = df_updated_f1.melt(id_vars=['Model', 'Type'], var_name='Saturation Level', value_name='Performance')

# Define the updated vibrant colors, including Ideal HDR
vibrant_colors_custom = {
    'Ideal HDR': '#6495ED',  # Cornflower Blue for Ideal HDR
    'α1.5': '#32CD32',  # Green
    'α2.0': '#FFD700',  # Yellow
    'α3.0': '#FF69B4',  # Pink
    'α4.0': '#FF6347',  # Red
}

# Define the order of models and types
models_order = ['YOLOv10s', 'YOLOv10l', 'YOLOv10x']
order = ['Ideal HDR', 'Recovery', 'Modulo', 'Saturate']

# Plotting with vibrant colors for updated F1-Score data and only one legend
fig, ax = plt.subplots(figsize=(15, 6))

# Bar plot with vibrant colors for updated F1-Score data
for i, model in enumerate(models_order):
    ax = plt.subplot(1, 3, i + 1)
    group = df_updated_f1_melted[df_updated_f1_melted['Model'] == model]
    group_pivot = group.pivot(index='Type', columns='Saturation Level', values='Performance')
    group_pivot = group_pivot.reindex(order[1:])  # Skip Ideal HDR for x-axis
    
    colors = [vibrant_colors_custom[col] for col in group_pivot.columns]
    group_pivot.plot(kind='bar', ax=ax, color=colors)
    
    ax.set_title(model)
    ax.set_ylabel('F1-Score')
    ax.set_xlabel('')
    ax.set_ylim(10, 65)
    ax.set_xticks(range(len(group_pivot.index)))
    ax.set_xticklabels(group_pivot.index, rotation=45)
    if i == 0:
        ax.legend(title='Saturation Level')
    else:
        ax.get_legend().remove()
    ax.set_yticks(range(10, 66, 5))

plt.tight_layout()
plt.show()





import pandas as pd
import matplotlib.pyplot as plt

# Updated data for Accuracy as provided by the user
updated_data_accuracy = {
    'Model': ['YOLOv10s', 'YOLOv10s', 'YOLOv10s', 'YOLOv10l', 'YOLOv10l', 'YOLOv10l', 'YOLOv10x', 'YOLOv10x', 'YOLOv10x'],
    'Type': ['Saturate', 'Modulo', 'Recovery', 'Saturate', 'Modulo', 'Recovery', 'Saturate', 'Modulo', 'Recovery'],
    'α1.5': [27.3, 31.5, 31.6, 34.3, 42.6, 42.0, 34.3, 44.1, 45.2],
    'α2.0': [25.6, 29.6, 32.0, 32.1, 42.9, 41.8, 31.4, 42.4, 45.1],
    'α3.0': [16.3, 26.7, 29.6, 23.3, 36.0, 36.1, 25.9, 41.2, 42.2],
    'α4.0': [11.5, 23.0, 26.0, 17.6, 28.5, 31.9, 19.3, 33.6, 36.2],
    'Ideal HDR': [32.1, 32.1, 32.1, 45.5, 45.5, 45.5, 50.2, 50.2, 50.2]
}

# Convert to DataFrame
df_updated_accuracy = pd.DataFrame(updated_data_accuracy)

# Melt the DataFrame for easier plotting
df_updated_accuracy_melted = df_updated_accuracy.melt(id_vars=['Model', 'Type'], var_name='Saturation Level', value_name='Performance')

# Define the updated vibrant colors, including Ideal HDR
vibrant_colors_custom = {
    'Ideal HDR': '#6495ED',  # Cornflower Blue for Ideal HDR
    'α1.5': '#32CD32',  # Green
    'α2.0': '#FFD700',  # Yellow
    'α3.0': '#FF69B4',  # Pink
    'α4.0': '#FF6347',  # Red
}

# Define the order of models and types
models_order = ['YOLOv10s', 'YOLOv10l', 'YOLOv10x']
order = ['Ideal HDR', 'Recovery', 'Modulo', 'Saturate']

# Plotting with vibrant colors for updated Accuracy data and only one legend
fig, ax = plt.subplots(figsize=(15, 6))

# Bar plot with vibrant colors for updated Accuracy data
for i, model in enumerate(models_order):
    ax = plt.subplot(1, 3, i + 1)
    group = df_updated_accuracy_melted[df_updated_accuracy_melted['Model'] == model]
    group_pivot = group.pivot(index='Type', columns='Saturation Level', values='Performance')
    group_pivot = group_pivot.reindex(order[1:])  # Skip Ideal HDR for x-axis
    
    colors = [vibrant_colors_custom[col] for col in group_pivot.columns]
    group_pivot.plot(kind='bar', ax=ax, color=colors)
    
    ax.set_title(model)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('')
    ax.set_ylim(10, 55)
    ax.set_xticks(range(len(group_pivot.index)))
    ax.set_xticklabels(group_pivot.index, rotation=45)
    if i == 0:
        ax.legend(title='Saturation Level')
    else:
        ax.get_legend().remove()
    ax.set_yticks(range(10, 56, 5))

plt.tight_layout()
plt.show()


