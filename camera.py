import gradio as gr
import cv2
from PIL import Image

def capture_image():
    # Abre la cámara web
    cap = cv2.VideoCapture(0)  # '0' es generalmente el valor predeterminado para la cámara web integrada
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return None
    
    # Captura un cuadro
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el cuadro de la cámara")
        return None
    
    # Convierte el cuadro a formato PIL para que Gradio pueda mostrarlo
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    
    # Libera la cámara
    cap.release()
    return pil_image

def app():
    with gr.Blocks() as demo:
        gr.Markdown("### Vista de Cámara Web en Vivo")
        
        # Botón para capturar imagen
        capture_button = gr.Button("Capturar Imagen")
        
        # Área de visualización de la imagen
        output_image = gr.Image(label="Imagen Capturada")

        # Acción al presionar el botón
        capture_button.click(
            fn=capture_image, 
            inputs=[],
            outputs=[output_image]
        )

    return demo

if __name__ == "__main__":
    app().launch()
