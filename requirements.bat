@echo off

:: Activar el entorno virtual
call conference111\Scripts\activate

:: Instalar PyTorch con soporte para CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

:: Instalar las demás dependencias
pip install -r requirements.txt
