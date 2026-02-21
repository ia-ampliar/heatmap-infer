# Usar Python 3.8 slim
FROM python:3.8-slim

# Instalar pacotes pré-compilados do OpenSlide e suas dependências
# 1. Instalar dependências do sistema necessárias
#    - libopenslide0: bibliotecas nativas do OpenSlide
#    - libglib2.0-0 : geralmente necessária pelo OpenSlide
#    - libsm6 e libxext6: bibliotecas comuns p/ OpenCV (evitam alguns erros de render)
#    - (Opcional) libtiff5, libjpeg62-turbo, libcairo2, etc. se precisar
RUN apt-get update && apt-get install -y \
    libopenslide0 \
    # python3-openslide \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libffi-dev \
    # Se quiser instalar as ferramentas CLI do OpenSlide, adicione:
    # openslide-tools \
    # (opcional) se precisar de Ferramentas no build ou debugging
    # build-essential gfortran
    && rm -rf /var/lib/apt/lists/*

# 2. Instalar bibliotecas Python
#    - boto3: para baixar do S3
#    - openslide-python: bindings do OpenSlide em Python
#    - opencv-python-headless: OpenCV sem libs de GUI (menor)
#    - numpy: às vezes já vem como dependência do opencv-python,
#      mas é bom explicitar
RUN pip install --no-cache-dir \
    boto3 \
    openslide-python \
    opencv-python-headless \
    tflite-runtime \
    numpy

# Cria um diretório de trabalho
WORKDIR /app

# Copia seu script
COPY main.py /app/
COPY tubular-bin-4x-v2.tflite /app/

# Comando de entrada (executa o script)
CMD ["python", "main.py"]

