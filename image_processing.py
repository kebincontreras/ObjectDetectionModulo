
import cv2
import numpy as np

def modulo(x, L):
    positive = x > 0
    x = x % L
    x = np.where( ( x == 0) &  positive, L, x)
    return x

def apply_blur(image, kernel_size):
    """Aplica desenfoque gaussiano a la imagen."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def clip_image(image, correction, sat_factor):
    """Clips image with saturation factor and correction."""
    processed_image = np.power(image, 1.0) * sat_factor*2
    clipped_image = np.clip(processed_image, 0, 1)
    return clipped_image

def wrap_image(image, correction, sat_factor):
    """Wraps image with saturation factor and correction."""
    processed_image = np.power(image, 1.0) * sat_factor
    wrapped_image =  modulo(processed_image, 1.0)
    return wrapped_image




import os
import matplotlib.pyplot as plt

def save_images(image_dir, image_id, original_image, clipped_image, wrapped_image, recon_image_np):
    # Crea una subcarpeta llamada 'processed_images' dentro de image_dir si no existe
    processed_dir = os.path.join(image_dir, "processed_images")
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    # Establece la ruta para guardar las imágenes procesadas
    save_path = os.path.join(processed_dir, f"{image_id}_processed.png")
    
    # Crea un subplot con 2x2
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    
    # Configura cada subplot
    axs[0, 0].imshow(original_image)
    axs[0, 0].set_title("Original")
    axs[0, 0].axis('off')

    axs[0, 1].imshow(clipped_image)
    axs[0, 1].set_title("Clipped")
    axs[0, 1].axis('off')

    axs[1, 0].imshow(wrapped_image)
    axs[1, 0].set_title("Wrapped")
    axs[1, 0].axis('off')

    axs[1, 1].imshow(recon_image_np)
    axs[1, 1].set_title("Reconstructed")
    axs[1, 1].axis('off')

    # Guarda la figura en la ruta especificada
    plt.savefig(save_path)
    plt.close(fig)  # Cierra la figura para liberar memoria

    print(f"Saved processed images to {save_path}")


